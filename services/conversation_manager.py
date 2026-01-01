"""
Conversation Manager - Session Context Tracking

Manages conversation context for follow-up queries by storing recent messages
and data context per session. Enables natural follow-up questions like 
"What about just Azure?" after showing all courses.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Single message in a conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sql_query: Optional[str] = None
    data_summary: Optional[str] = None


@dataclass
class ConversationContext:
    """Context for a conversation session"""
    conversation_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    current_sql: Optional[str] = None
    current_data_summary: Optional[str] = None
    relevant_tables: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)


class ConversationManager:
    """
    Manages conversation contexts for the analytics agent.
    
    Features:
    - Store conversation history per session
    - Track current data context (SQL, results summary)
    - Provide context for follow-up queries
    - Auto-cleanup of stale sessions
    """
    
    def __init__(self, max_messages: int = 10, session_timeout_hours: int = 2):
        """
        Initialize the conversation manager.
        
        Args:
            max_messages: Maximum messages to keep per conversation
            session_timeout_hours: Hours after which inactive sessions are cleaned
        """
        self.conversations: Dict[str, ConversationContext] = {}
        self.max_messages = max_messages
        self.session_timeout = timedelta(hours=session_timeout_hours)
        logger.info(f"ConversationManager initialized (max_messages={max_messages})")
    
    def get_or_create_conversation(self, conversation_id: Optional[str] = None) -> ConversationContext:
        """
        Get existing conversation or create new one.
        
        Args:
            conversation_id: Optional ID to look up or create
            
        Returns:
            ConversationContext for the session
        """
        # Clean up stale sessions periodically
        self._cleanup_stale_sessions()
        
        if conversation_id and conversation_id in self.conversations:
            context = self.conversations[conversation_id]
            context.last_active = datetime.now()
            return context
        
        # Create new conversation
        new_id = conversation_id or str(uuid.uuid4())
        context = ConversationContext(conversation_id=new_id)
        self.conversations[new_id] = context
        logger.info(f"Created new conversation: {new_id}")
        return context
    
    def add_user_message(self, conversation_id: str, content: str) -> None:
        """Add a user message to the conversation."""
        context = self.get_or_create_conversation(conversation_id)
        message = ConversationMessage(role="user", content=content)
        context.messages.append(message)
        self._trim_messages(context)
        logger.debug(f"Added user message to {conversation_id}: {content[:50]}...")
    
    def add_assistant_message(
        self, 
        conversation_id: str, 
        content: str,
        sql_query: Optional[str] = None,
        data_summary: Optional[str] = None
    ) -> None:
        """Add an assistant message with optional SQL and data context."""
        context = self.get_or_create_conversation(conversation_id)
        message = ConversationMessage(
            role="assistant", 
            content=content,
            sql_query=sql_query,
            data_summary=data_summary
        )
        context.messages.append(message)
        
        # Update current context
        if sql_query:
            context.current_sql = sql_query
        if data_summary:
            context.current_data_summary = data_summary
            
        self._trim_messages(context)
        logger.debug(f"Added assistant message to {conversation_id}")
    
    def update_data_context(
        self, 
        conversation_id: str, 
        sql_query: str,
        data_summary: str,
        tables_used: List[str]
    ) -> None:
        """Update the current data context for the conversation."""
        context = self.get_or_create_conversation(conversation_id)
        context.current_sql = sql_query
        context.current_data_summary = data_summary
        context.relevant_tables = tables_used
        context.last_active = datetime.now()
    
    def get_context_for_llm(self, conversation_id: str) -> str:
        """
        Generate a context string for the LLM to understand conversation history.
        
        Returns:
            Formatted context string for inclusion in LLM prompts
        """
        context = self.conversations.get(conversation_id)
        if not context or not context.messages:
            return ""
        
        # Build context string
        parts = []
        
        # Add recent messages (skip thinking/system messages)
        recent_exchanges = []
        for msg in context.messages[-6:]:  # Last 6 messages
            role_label = "User" if msg.role == "user" else "Assistant"
            recent_exchanges.append(f"{role_label}: {msg.content[:200]}")
        
        if recent_exchanges:
            parts.append("RECENT CONVERSATION:")
            parts.extend(recent_exchanges)
        
        # Add current data context if available
        if context.current_sql:
            parts.append(f"\nLAST SQL QUERY:\n{context.current_sql}")
        
        if context.current_data_summary:
            parts.append(f"\nLAST RESULT SUMMARY:\n{context.current_data_summary}")
        
        if context.relevant_tables:
            parts.append(f"\nRELEVANT TABLES: {', '.join(context.relevant_tables)}")
        
        return "\n".join(parts)
    
    def is_follow_up_query(self, conversation_id: str, user_query: str) -> bool:
        """
        Detect if the query is a follow-up to previous context.
        
        Args:
            conversation_id: Session ID
            user_query: Current user query
            
        Returns:
            True if this appears to be a follow-up question
        """
        context = self.conversations.get(conversation_id)
        if not context or len(context.messages) < 2:
            return False
        
        # Follow-up indicators
        follow_up_phrases = [
            "what about", "how about", "and what", "show me",
            "the same", "those", "these", "them", "it", "that",
            "but", "instead", "now", "also", "compare", "vs",
            "last", "previous", "before", "more", "less", "just"
        ]
        
        query_lower = user_query.lower()
        return any(phrase in query_lower for phrase in follow_up_phrases)
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation for debugging/display."""
        context = self.conversations.get(conversation_id)
        if not context:
            return {"exists": False}
        
        return {
            "exists": True,
            "conversation_id": conversation_id,
            "message_count": len(context.messages),
            "has_data_context": context.current_sql is not None,
            "relevant_tables": context.relevant_tables,
            "created_at": context.created_at.isoformat(),
            "last_active": context.last_active.isoformat()
        }
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a specific conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation: {conversation_id}")
            return True
        return False
    
    def _trim_messages(self, context: ConversationContext) -> None:
        """Trim messages to max_messages limit."""
        if len(context.messages) > self.max_messages:
            context.messages = context.messages[-self.max_messages:]
    
    def _cleanup_stale_sessions(self) -> None:
        """Remove sessions that have been inactive beyond timeout."""
        now = datetime.now()
        stale_ids = [
            cid for cid, ctx in self.conversations.items()
            if (now - ctx.last_active) > self.session_timeout
        ]
        for cid in stale_ids:
            del self.conversations[cid]
            logger.info(f"Cleaned up stale session: {cid}")


# Singleton instance
conversation_manager = ConversationManager()
