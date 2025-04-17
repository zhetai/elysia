from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class QueryData(BaseModel):
    user_id: str
    conversation_id: str
    query_id: str
    query: str
    collection_names: list[str]
    route: Optional[str] = ""
    mimick: Optional[bool] = False


class GetCollectionsData(BaseModel):
    user_id: str


class ViewPaginatedCollectionData(BaseModel):
    user_id: str
    collection_name: str
    page_size: int
    page_number: int
    sort_on: Optional[str] = None
    ascending: Optional[bool] = False
    filter_config: Optional[dict[str, Any]] = {}


class InitialiseTreeData(BaseModel):
    user_id: str
    conversation_id: str
    debug: Optional[bool] = False


class NERData(BaseModel):
    text: str


class TitleData(BaseModel):
    user_id: str
    conversation_id: str
    text: str


class SetCollectionsData(BaseModel):
    user_id: str
    conversation_id: str
    collection_names: List[str]


class GetObjectData(BaseModel):
    user_id: str
    collection_name: str
    uuid: str


class ObjectRelevanceData(BaseModel):
    user_id: str
    conversation_id: str
    query_id: str
    objects: list[dict]


class ProcessCollectionData(BaseModel):
    user_id: str
    collection_name: str


class DebugData(BaseModel):
    conversation_id: str
    user_id: str


class CollectionMetadataData(BaseModel):
    user_id: str
    collection_name: str


class UserCollectionsData(BaseModel):
    user_id: str


class AddFeedbackData(BaseModel):
    user_id: str
    conversation_id: str
    query_id: str
    feedback: int


class RemoveFeedbackData(BaseModel):
    user_id: str
    conversation_id: str
    query_id: str


class GetUserRequestsData(BaseModel):
    user_id: str


class FeedbackMetadataData(BaseModel):
    user_id: str


class InstantReplyData(BaseModel):
    user_id: str
    user_prompt: str


class FollowUpSuggestionsData(BaseModel):
    user_id: str
    conversation_id: str


class DefaultConfigData(BaseModel):
    user_id: str


class ChangeConfigData(BaseModel):
    user_id: str
    config: dict[str, Any]


class LoadConfigData(BaseModel):
    user_id: str
    filepath: str


class SaveConfigData(BaseModel):
    user_id: str
    filepath: str
