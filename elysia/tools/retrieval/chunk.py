import asyncio

import spacy
from tqdm.auto import tqdm
from weaviate.classes.config import Configure, DataType, Property, ReferenceProperty
from weaviate.collections.classes.data import DataObject, DataReference
from weaviate.exceptions import WeaviateInvalidInputError
from weaviate.util import generate_uuid5

from elysia.util.client import ClientManager
from elysia.util.collection import (
    async_get_collection_weaviate_data_types,
    get_collection_weaviate_data_types,
)


class Chunker:
    def __init__(
        self,
        chunking_strategy: str = "fixed",
        num_tokens: int = 256,
        num_sentences: int = 5,
    ):
        self.chunking_strategy = chunking_strategy
        assert chunking_strategy in ["fixed", "sentences"]

        self.nlp = spacy.load("en_core_web_sm")

        self.num_tokens = num_tokens
        self.num_sentences = num_sentences

    def count_tokens(self, document):
        return len(self.nlp(document))

    def chunk_by_sentences(self, document, num_sentences=None, overlap_sentences=1):
        """
        Given a document (string), return the sentences as chunks and span annotations (start and end indices of chunks).
        Using spaCy to do sentence chunking.
        """
        if num_sentences is None:
            num_sentences = self.num_sentences

        if overlap_sentences >= num_sentences:
            print(
                f"Warning: overlap_sentences ({overlap_sentences}) is greater than num_sentences ({num_sentences}). Setting overlap to {num_sentences - 1}"
            )
            overlap_sentences = num_sentences - 1

        doc = self.nlp(document)
        sentences = list(doc.sents)  # Get sentence boundaries from spaCy

        span_annotations = []
        chunks = []

        i = 0
        while i < len(sentences):
            # Get chunk of num_sentences sentences
            chunk_sentences = sentences[i : i + num_sentences]
            if not chunk_sentences:
                break

            # Get start and end char positions
            start_char = chunk_sentences[0].start_char
            end_char = chunk_sentences[-1].end_char

            # Add chunk and its span annotation
            chunks.append(document[start_char:end_char])
            span_annotations.append((start_char, end_char))

            # Move forward but account for overlap
            i += num_sentences - overlap_sentences

        return chunks, span_annotations

    def chunk_by_tokens(self, document, num_tokens=None, overlap_tokens=32):
        """
        Given a document (string), return the tokens as chunks and span annotations (start and end indices of chunks).
        Includes overlapping tokens between chunks for better context preservation.
        Uses spaCy for tokenization.
        """
        if num_tokens is None:
            num_tokens = self.num_tokens

        doc = self.nlp(document)
        tokens = list(doc)  # Get tokens from spaCy doc

        span_annotations = []
        chunks = []
        i = 0

        while i < len(tokens):
            # Find end index for current chunk
            end_idx = min(i + num_tokens, len(tokens))
            chunk_tokens = tokens[i:end_idx]

            # Get character spans for the chunk
            start_char = chunk_tokens[0].idx
            end_char = chunk_tokens[-1].idx + len(chunk_tokens[-1])

            # Add chunk and its span annotation
            chunks.append(document[start_char:end_char])
            span_annotations.append((start_char, end_char))

            # Move forward but account for overlap
            i += max(1, num_tokens - overlap_tokens)

        return chunks, span_annotations

    def chunk(self, document):
        if self.chunking_strategy == "sentences":
            return self.chunk_by_sentences(document)
        elif self.chunking_strategy == "tokens":
            return self.chunk_by_tokens(document)


class AsyncCollectionChunker:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.chunker = Chunker("sentences", num_sentences=5)

    async def create_chunked_reference(
        self, content_field: str, client_manager: ClientManager
    ):
        async with client_manager.connect_to_async_client() as client:
            chunked_collection = await self.get_chunked_collection(
                content_field, client
            )
            full_collection = client.collections.get(self.collection_name)

            # add reference to chunked collection
            try:
                await chunked_collection.config.add_reference(
                    ReferenceProperty(
                        name="fullDocument", target_collection=full_collection.name
                    )
                )
            except WeaviateInvalidInputError:
                # already exists
                pass

            # add reference to full collection
            try:
                await full_collection.config.add_reference(
                    ReferenceProperty(
                        name="isChunked", target_collection=chunked_collection.name
                    )
                )
            except WeaviateInvalidInputError:
                # already exists
                pass

    def get_chunked_collection_name(self):
        return f"ELYSIA_CHUNKED_{self.collection_name.lower()}__"

    async def chunked_collection_exists(self, client):
        return await client.collections.exists(self.get_chunked_collection_name())

    async def get_chunked_collection(self, content_field: str, client):
        # get properties of main collection
        data_types = await async_get_collection_weaviate_data_types(
            client, self.collection_name
        )

        if await client.collections.exists(self.get_chunked_collection_name()):
            return client.collections.get(self.get_chunked_collection_name())
        else:
            return await client.collections.create(
                self.get_chunked_collection_name(),
                properties=[
                    Property(name=content_field, data_type=DataType.TEXT),
                    Property(
                        name="chunk_spans",
                        data_type=DataType.INT_ARRAY,
                        nested_data_type=DataType.INT,
                    ),
                ],
                references=[
                    ReferenceProperty(
                        name="fullDocument", target_collection=self.collection_name
                    )
                ],
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-large",
                    dimensions=256,  # matryoshka embedding
                ),
                vector_index_config=Configure.VectorIndex.hnsw(
                    quantizer=Configure.VectorIndex.Quantizer.sq()  # scalar quantization
                ),
            )

    def generate_uuids(
        self,
        chunks: list[str],
        spans: list[tuple[int, int]],
        content_field: str,
    ):
        chunked_uuids = []
        for i, (chunk, span) in enumerate(zip(chunks, spans)):
            data_object = {content_field: chunk, "chunk_spans": span}
            chunked_uuids.append(generate_uuid5(data_object))
        return chunked_uuids

    async def chunk_single_object(self, object, content_field: str):
        chunks, spans = self.chunker.chunk(object.properties[content_field])
        chunk_uuids = self.generate_uuids(chunks, spans, content_field)
        return chunks, spans, chunk_uuids

    async def chunk_objects_parallel(
        self, unchunked_objects, unchunked_uuids, content_field: str
    ):
        tasks = [
            self.chunk_single_object(obj, content_field) for obj in unchunked_objects
        ]

        results = await asyncio.gather(*tasks)

        original_uuid_to_chunks = {}
        original_uuid_to_spans = {}
        original_uuid_to_chunk_uuids = {}

        for uuid, (chunks, spans, chunk_uuids) in zip(unchunked_uuids, results):
            original_uuid_to_chunks[uuid] = chunks
            original_uuid_to_spans[uuid] = spans
            original_uuid_to_chunk_uuids[uuid] = chunk_uuids

        return (
            original_uuid_to_chunks,
            original_uuid_to_spans,
            original_uuid_to_chunk_uuids,
        )

    async def insert_chunks(
        self,
        chunked_collection,
        original_uuid_to_chunks,
        original_uuid_to_spans,
        original_uuid_to_chunk_uuids,
        content_field: str,
    ):
        data_objects = []
        for original_uuid in original_uuid_to_chunks:
            for i, (chunk, span, uuid) in enumerate(
                zip(
                    original_uuid_to_chunks[original_uuid],
                    original_uuid_to_spans[original_uuid],
                    original_uuid_to_chunk_uuids[original_uuid],
                )
            ):
                data_objects.append(
                    DataObject(
                        properties={content_field: chunk, "chunk_spans": span},
                        uuid=uuid,
                        references={"fullDocument": original_uuid},
                    )
                )

        batch_size = 5
        data_objects_batches = [
            data_objects[i : i + batch_size]
            for i in range(0, len(data_objects), batch_size)
        ]

        tasks = [
            chunked_collection.data.insert_many(data_objects)
            for data_objects in data_objects_batches
        ]
        await asyncio.gather(*tasks)

    async def insert_references(self, full_collection, original_uuid_to_chunk_uuids):
        references = []
        for original_uuid in original_uuid_to_chunk_uuids:
            for chunk_uuid in original_uuid_to_chunk_uuids[original_uuid]:
                references.append(
                    DataReference(
                        from_uuid=original_uuid,
                        from_property="isChunked",
                        to_uuid=chunk_uuid,
                    )
                )

        batch_size = 5
        references_batches = [
            references[i : i + batch_size]
            for i in range(0, len(references), batch_size)
        ]

        tasks = [
            full_collection.data.reference_add_many(references)
            for references in references_batches
        ]
        await asyncio.gather(*tasks)

    def get_chunked_objects(self, objects):
        """
        Given a list of weaviate objects, find if a reference exists to a chunked object in the separate collection.
        Return the UUIDs of `objects` that have a reference to a chunked object (already have been chunked).
        """
        uuids = []
        for object in objects.objects:
            if (
                object.references is not None
                and "isChunked" in object.references
                and len(object.references["isChunked"].objects) > 0
                and any(
                    [
                        o.collection == self.get_chunked_collection_name()
                        for o in object.references["isChunked"].objects
                    ]
                )
            ):
                uuids.append(str(object.uuid))
        return uuids

    async def __call__(
        self, objects, content_field: str, client_manager: ClientManager
    ):
        # always create the chunked reference, if it already exists, it will be skipped
        # self.create_chunked_reference(content_field)

        # get all UUIDs of current objects
        all_uuids = [str(object.uuid) for object in objects.objects]

        # get all UUIDs of chunked objects
        chunked_uuids = self.get_chunked_objects(objects)

        # get all UUIDs of unchunked objects - these are the objects that need to be chunked
        unchunked_uuids = [
            str(uuid) for uuid in all_uuids if str(uuid) not in chunked_uuids
        ]

        # get all unchunked objects for chunking and inserting
        unchunked_objects = [
            object for object in objects.objects if str(object.uuid) in unchunked_uuids
        ]

        assert len(unchunked_objects) == len(unchunked_uuids)

        # if there are unchunked objects, chunk them
        if len(unchunked_objects) > 0:
            async with client_manager.connect_to_async_client() as client:
                # Get collections
                chunked_collection = await self.get_chunked_collection(
                    content_field, client
                )
                full_collection = client.collections.get(self.collection_name)

                (
                    original_uuid_to_chunks,
                    original_uuid_to_spans,
                    original_uuid_to_chunk_uuids,
                ) = await self.chunk_objects_parallel(
                    unchunked_objects, unchunked_uuids, content_field
                )

                # insert into weaviate
                await self.insert_chunks(
                    chunked_collection,
                    original_uuid_to_chunks,
                    original_uuid_to_spans,
                    original_uuid_to_chunk_uuids,
                    content_field,
                )
                await self.insert_references(
                    full_collection, original_uuid_to_chunk_uuids
                )


# class SyncCollectionChunker:
#     def __init__(self, collection_name):
#         self.collection_name = collection_name
#         self.chunker = Chunker("sentences", num_sentences=5)

#     def create_chunked_reference(
#         self, content_field: str, client_manager: ClientManager
#     ):
#         with client_manager.connect_to_client() as client:
#             collection = client.collections.get(
#                 f"ELYSIA_CHUNKED_{self.collection_name}__"
#             )
#             try:
#                 collection.config.add_reference(
#                     ReferenceProperty(
#                         name="isChunked",
#                         target_collection=f"ELYSIA_CHUNKED_{self.collection_name}__",
#                     )
#                 )
#             except WeaviateInvalidInputError:
#                 pass

#     def chunked_collection_exists(self, client):
#         return client.collections.exists(f"ELYSIA_CHUNKED_{self.collection_name}__")

#     def get_chunked_collection(self, content_field: str, client):
#         # get properties of main collection
#         data_types = get_collection_weaviate_data_types(client, self.collection_name)

#         if client.collections.exists(f"ELYSIA_CHUNKED_{self.collection_name}__"):
#             return client.collections.get(f"ELYSIA_CHUNKED_{self.collection_name}__")
#         else:
#             return client.collections.create(
#                 f"ELYSIA_CHUNKED_{self.collection_name}__",
#                 properties=[
#                     Property(name=content_field, data_type=DataType.TEXT),
#                     Property(
#                         name="chunk_spans",
#                         data_type=DataType.INT_ARRAY,
#                         nested_data_type=DataType.INT,
#                     ),
#                 ]
#                 + [
#                     Property(name=name, data_type=data_type)
#                     for name, data_type in data_types.items()
#                     if name != content_field
#                 ],  # add all properties except the content field TODO: this should be a reference to the original so no need to store twice
#                 references=[
#                     ReferenceProperty(
#                         name="fullDocument", target_collection=self.collection_name
#                     )
#                 ],
#                 vectorizer_config=Configure.Vectorizer.text2vec_openai(
#                     model="text-embedding-3-large", dimensions=256
#                 ),
#                 vector_index_config=Configure.VectorIndex.hnsw(
#                     quantizer=Configure.VectorIndex.Quantizer.sq()  # scalar quantization
#                 ),
#             )

#     def generate_uuids(
#         self,
#         chunks: list[str],
#         spans: list[tuple[int, int]],
#         properties,
#         content_field: str,
#     ):
#         chunked_uuids = []
#         for i, (chunk, span) in enumerate(zip(chunks, spans)):
#             data_object = {content_field: chunk, "chunk_spans": span, **properties}
#             chunked_uuids.append(generate_uuid5(data_object))
#         return chunked_uuids

#     def batch_sync_insert_chunks(
#         self,
#         chunked_collection,
#         chunks,
#         spans,
#         chunked_uuids,
#         properties,
#         content_field: str,
#     ):
#         with chunked_collection.batch.dynamic() as batch:
#             for i, (chunk, span) in enumerate(zip(chunks, spans)):
#                 data_object = {
#                     content_field: chunk,
#                     "chunk_spans": span,
#                     **properties[i],
#                 }
#                 batch.add_object(data_object, uuid=chunked_uuids[i])

#     def batch_sync_insert_references(
#         self, chunked_collection, full_collection, both_uuids
#     ):
#         """
#         Both UUIDS: {original_uuid: [chunked_uuids], ...}
#         keys are the original and list is the chunked UUIDs corresponding to the original
#         """
#         with chunked_collection.batch.dynamic() as chunked_batch:
#             for original_uuid in both_uuids:
#                 for chunked_uuid in both_uuids[original_uuid]:
#                     chunked_batch.add_reference(
#                         from_uuid=chunked_uuid,
#                         from_property="fullDocument",
#                         to=original_uuid,
#                     )

#         with full_collection.batch.dynamic() as full_batch:
#             for original_uuid in both_uuids:
#                 for chunked_uuid in both_uuids[original_uuid]:
#                     full_batch.add_reference(
#                         from_uuid=original_uuid,
#                         from_property="isChunked",
#                         to=chunked_uuid,
#                     )

#     def get_chunked_objects(self, objects):
#         """
#         Given a list of weaviate objects, find if a reference exists to a chunked object in the separate collection.
#         Return the UUIDs of `objects` that have a reference to a chunked object.
#         """
#         uuids = []
#         for object in objects.objects:
#             if (
#                 object.references is not None
#                 and "isChunked" in object.references
#                 and len(object.references["isChunked"].objects) > 0
#             ):
#                 uuids.append(object.uuid)
#         return uuids

#     def __call__(self, objects, content_field: str, client_manager: ClientManager):
#         # always create the chunked reference, if it already exists, it will be skipped
#         # self.create_chunked_reference(content_field)

#         # get all UUIDs of current objects
#         all_uuids = [object.uuid for object in objects.objects]

#         # get all UUIDs of chunked objects
#         chunked_uuids = self.get_chunked_objects(objects)

#         # get all UUIDs of unchunked objects - these are the objects that need to be chunked
#         unchunked_uuids = [uuid for uuid in all_uuids if uuid not in chunked_uuids]

#         # get all unchunked objects for chunking and inserting
#         unchunked_objects = [
#             object for object in objects.objects if object.uuid in unchunked_uuids
#         ]

#         assert len(unchunked_objects) == len(unchunked_uuids)

#         # if there are unchunked objects, chunk them
#         if len(unchunked_objects) > 0:
#             with client_manager.connect_to_client() as client:
#                 # Get collections
#                 chunked_collection = self.get_chunked_collection(content_field, client)
#                 full_collection = client.collections.get(self.collection_name)

#                 # keeping track of the chunks/spans, output uuids for each chunk object, and other properties from the collection
#                 all_chunks = []
#                 all_spans = []
#                 chunk_uuids = []
#                 all_uuids = {}
#                 all_properties = []
#                 for i, object in enumerate(unchunked_objects):
#                     # chunk the text and get spans
#                     chunks, spans = self.chunker.chunk(object.properties[content_field])

#                     # get other properties from the collection for this object
#                     other_properties = {
#                         name: object.properties[name]
#                         for name in object.properties.keys()
#                         if name != content_field
#                     }
#                     chunk_uuids_i = self.generate_uuids(
#                         chunks, spans, other_properties, content_field
#                     )

#                     # a single dict of properties for all chunks corresponding to the same original object
#                     all_properties.extend([other_properties] * len(chunks))

#                     # all chunks and spans
#                     all_chunks.extend(chunks)
#                     all_spans.extend(spans)

#                     # chunk_uuids (list of uuids corresponding to all_chunks), all_uuids (dict of original uuids to chunked uuids, mapping)
#                     chunk_uuids.extend(chunk_uuids_i)
#                     all_uuids[unchunked_uuids[i]] = chunk_uuids_i

#                 # insert into weaviate
#                 self.batch_sync_insert_chunks(
#                     chunked_collection,
#                     all_chunks,
#                     all_spans,
#                     chunk_uuids,
#                     all_properties,
#                     content_field,
#                 )
#                 self.batch_sync_insert_references(
#                     chunked_collection, full_collection, all_uuids
#                 )
