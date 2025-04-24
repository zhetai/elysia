import asyncio
import os
import unittest
from typing import Any

from elysia.objects import Completed, Error, Response, Status, Text, Update, Warning
from elysia.tools.objects import (
    BoringGeneric,
    Conversation,
    Document,
    Ecommerce,
    EpicGeneric,
    Message,
    Ticket,
)
from elysia.tools.retrieval.chunk import AsyncCollectionChunker
from elysia.tools.retrieval.objects import (
    BoringGenericRetrieval,
    ConversationRetrieval,
    DocumentRetrieval,
    EcommerceRetrieval,
    EpicGenericRetrieval,
    MessageRetrieval,
    TicketRetrieval,
)
from elysia.util.client import ClientManager
from elysia.util.parsing import format_dict_to_serialisable


# TODO: add text
class TestObjects(unittest.TestCase):
    def get_event_loop(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop

    async def start_client_manager(self):
        client_manager = ClientManager(
            wcd_url=os.getenv("WCD_URL"),
            wcd_api_key=os.getenv("WCD_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        await client_manager.start_clients()
        return client_manager

    async def do_basic_result(self, object_type: Any, object_name: str):
        result = object_type(
            objects=[{"a": 1, "b": 2, "c": 3}],
            metadata={"metadata_a": 1, "metadata_b": 2, "metadata_c": 3},
        )
        result_json = result.to_json()

        self.assertTrue(
            result_json == [{"a": 1, "b": 2, "c": 3}],
            "Objects should be equal",
        )
        frontend_result = await result.to_frontend(
            "user_id", "conversation_id", "query_id"
        )

        # all id vars should be present
        for key in ["user_id", "conversation_id", "query_id"]:
            self.assertIn(
                key,
                frontend_result,
                f"'{key}' not found in the frontend result",
            )

        # type and payload type should be correct
        self.assertIs(frontend_result["type"], "result")
        self.assertIs(frontend_result["payload"]["type"], object_name)

        # objects should be correct (but there can be more objects in frontend result)
        for i, obj in enumerate(result_json):
            for key in obj:
                self.assertIn(
                    key,
                    frontend_result["payload"]["objects"][i],
                    f"'{key}' not found in the frontend result",
                )

        # metadata should be the same
        self.assertIs(frontend_result["payload"]["metadata"], result.metadata)

    async def do_result_with_mapping(
        self,
        object_type: Any,
        objects: list[dict],
        mapping: dict,
        unmapped_keys: list[str] = [],
    ):
        result = object_type(
            objects=objects, metadata={}, mapping=mapping, unmapped_keys=unmapped_keys
        )
        result_json = result.to_json(True)

        for i, obj in enumerate(objects):
            for mapping_key in mapping:
                self.assertIn(mapping_key, result_json[i])

            for obj_key in obj:
                if obj_key not in unmapped_keys:
                    self.assertNotIn(obj_key, result_json[i])

            for unmapped_key in unmapped_keys:
                self.assertIn(unmapped_key, result_json[i])

        frontend_result = await result.to_frontend(
            "user_id", "conversation_id", "query_id"
        )

        for i, obj in enumerate(objects):
            for mapping_key in mapping:
                self.assertIn(mapping_key, frontend_result["payload"]["objects"][i])

    async def do_update(self, object_type: Any, object_name: str):
        result = object_type(
            "this is a test",
        )
        frontend_result = await result.to_frontend(
            "user_id",
            "conversation_id",
        )
        self.assertIs(frontend_result["type"], object_name)
        self.assertIsInstance(frontend_result["payload"], dict)

    def test_basic_results(self):
        loop = self.get_event_loop()
        types = [
            BoringGeneric,
            EpicGeneric,
            Document,
            Ticket,
            Ecommerce,
            Message,
            Conversation,
        ]
        names = [
            "boring_generic",
            "epic_generic",
            "document",
            "ticket",
            "ecommerce",
            "message",
            "conversation",
        ]
        for object_type, object_name in zip(types, names):
            loop.run_until_complete(self.do_basic_result(object_type, object_name))

    def test_result_with_mapping(self):
        for object_type in [
            EpicGeneric,
            Document,
            Ticket,
            Ecommerce,
            Message,
            Conversation,
        ]:
            loop = self.get_event_loop()
            loop.run_until_complete(
                self.do_result_with_mapping(
                    object_type,
                    [
                        {
                            "card_text": "this is card text",
                            "card_value": 2,
                            "card_id": "123",
                        },
                        {
                            "card_text": "this is card text 2",
                            "card_value": 3,
                            "card_id": "456",
                        },
                    ],
                    {"text": "card_text", "value": "card_value"},
                    ["card_id"],
                )
            )

    def test_updates(self):
        loop = self.get_event_loop()
        types = [
            Status,
            Warning,
            Error,
        ]
        names = [
            "status",
            "warning",
            "error",
        ]
        for object_type, object_name in zip(types, names):
            loop.run_until_complete(self.do_update(object_type, object_name))

    async def _retrieve_documents(
        self, collection_name: str, client_manager: ClientManager
    ):
        async with client_manager.connect_to_async_client() as client:
            collection = client.collections.get(collection_name)
            documents = await collection.query.fetch_objects(limit=2)
        return documents

    async def _chunk_documents(
        self, collection_name: str, response, client_manager: ClientManager
    ):
        collection_chunker = AsyncCollectionChunker(collection_name)
        await collection_chunker.create_chunked_reference("content", client_manager)
        await collection_chunker(response, "content", client_manager)

    async def get_collection_information(
        self, collection_name: str, client_manager: ClientManager
    ):
        async with client_manager.connect_to_async_client() as client:
            metadata_name = f"ELYSIA_METADATA_{collection_name.lower()}__"
            metadata_collection = client.collections.get(metadata_name)
            metadata = await metadata_collection.query.fetch_objects(limit=1)
            properties = metadata.objects[0].properties
            format_dict_to_serialisable(properties)
            return properties

    # TODO: test more retrieval types
    async def get_document_retrievals(self):
        collection_name = "ml_wikipedia"
        client_manager = await self.start_client_manager()

        # get the collection information
        collection_information = await self.get_collection_information(
            collection_name, client_manager
        )

        # retrieve some documents
        unchunked_documents = await self._retrieve_documents(
            collection_name, client_manager
        )

        # chunk the documents
        await self._chunk_documents(
            collection_name, unchunked_documents, client_manager
        )

        # retrieve the chunked documents
        chunked_documents = await self._retrieve_documents(
            f"ELYSIA_CHUNKED_{collection_name.lower()}__", client_manager
        )

        chunked_document_properties = [
            doc.properties for doc in chunked_documents.objects
        ]
        unchunked_document_properties = [
            doc.properties for doc in unchunked_documents.objects
        ]

        for i, prop in enumerate(chunked_document_properties):
            prop["uuid"] = chunked_documents.objects[i].uuid.hex

        for i, prop in enumerate(unchunked_document_properties):
            prop["uuid"] = unchunked_documents.objects[i].uuid.hex

        # create a retrieval object
        non_chunked_retrieval = DocumentRetrieval(
            objects=[doc.properties for doc in unchunked_documents.objects],
            metadata={"collection_name": "ml_wikipedia", "chunked": False},
            mapping=collection_information["mappings"]["document"],
        )
        chunked_retrieval = DocumentRetrieval(
            objects=[doc.properties for doc in chunked_documents.objects],
            metadata={"collection_name": "ml_wikipedia", "chunked": True},
            mapping=collection_information["mappings"]["document"],
        )
        await chunked_retrieval.async_init(client_manager)
        await client_manager.close_clients()

        return non_chunked_retrieval, chunked_retrieval

    def test_document_retrieval(self):
        loop = self.get_event_loop()
        non_chunked_retrieval, chunked_retrieval = loop.run_until_complete(
            self.get_document_retrievals()
        )

        non_chunked_json = non_chunked_retrieval.to_json(mapping=True)
        chunked_json = chunked_retrieval.to_json(mapping=False)

        chunked_fe = loop.run_until_complete(
            chunked_retrieval.to_frontend("user_id", "conversation_id", "query_id")
        )

        # Check that the document attached to the chunk is >= length of the chunk
        for i, full_doc_obj in enumerate(chunked_fe["payload"]["objects"]):
            self.assertTrue(
                len(full_doc_obj["content"]) >= len(chunked_json[i]["content"])
            )

        # Check that all chunk spans from the chunk exist in the full documnet
        all_chunk_spans = [(obj["uuid"], obj["chunk_spans"]) for obj in chunked_json]

        # Find all unique instances of chunkID+chunkSpan
        which_chunk_spans_exist = {
            str(all_chunk_spans[i][0]) + str(all_chunk_spans[i][1]): False
            for i in range(len(all_chunk_spans))
        }

        # Look through full documents (FE payload-objects) and check if the chunk content is present
        for i, full_doc_obj in enumerate(chunked_fe["payload"]["objects"]):
            for chunk_spans in full_doc_obj["chunk_spans"]:
                for span in chunk_spans:
                    ind = str(span["uuid"]) + str([span["start"], span["end"]])
                    which_chunk_spans_exist[ind] = True

        self.assertTrue(all(which_chunk_spans_exist.values()))


if __name__ == "__main__":
    unittest.main()
    # TestObjects().test_document_retrieval()
