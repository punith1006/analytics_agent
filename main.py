"""
LMS Analytics Agent - FastAPI Application

Main application entry point with SSE endpoint for real-time chat streaming.
Orchestrates the multi-agent workflow: Schema Discovery → SQL Generation → 
Execution → Analysis → Visualization.
"""
import json
import logging
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import settings
from services.schema_service import SchemaService
from services.sql_executor import SQLExecutor, SQLValidationError, SQLExecutionError
from services.llm_service import LLMService
from services.chart_generator import ChartGenerator
from services.conversation_manager import conversation_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
schema_service = SchemaService()
sql_executor = SQLExecutor()
llm_service = LLMService()
chart_generator = ChartGenerator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown"""
    # Startup: Pre-load schema cache
    logger.info("Starting LMS Analytics Agent...")
    try:
        schema = schema_service.get_schema()
        logger.info(f"Schema loaded: {len(schema['tables'])} tables discovered")
    except Exception as e:
        logger.error(f"Failed to load schema: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LMS Analytics Agent...")


# Create FastAPI app
app = FastAPI(
    title="LMS Analytics Agent",
    description="AI-powered data analytics agent for Learning Management System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model"""
    query: str
    conversation_id: Optional[str] = None


class SchemaResponse(BaseModel):
    """Schema response model"""
    database: str
    table_count: int
    tables: dict


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "LMS Analytics Agent"}


# Schema endpoint for debugging
@app.get("/api/schema")
async def get_schema():
    """Get database schema (for debugging)"""
    try:
        schema = schema_service.get_schema()
        return {
            "database": schema["database"],
            "table_count": len(schema["tables"]),
            "tables": list(schema["tables"].keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main chat endpoint with SSE
@app.post("/api/analytics/chat")
async def analytics_chat(request: ChatRequest):
    """
    Main analytics chat endpoint using Server-Sent Events (SSE)
    
    Streams responses in the following event types:
    - thinking: Agent is processing
    - sql_generated: SQL query was generated
    - data_retrieved: Query executed, data retrieved
    - analysis: Insights and analysis generated
    - visualization: Chart configuration ready
    - complete: Processing complete
    - error: Error occurred
    """
    async def event_generator():
        try:
            user_query = request.query
            conversation_id = request.conversation_id or "default"
            logger.info(f"Processing query: {user_query} (conversation: {conversation_id})")
            
            # Track user message in conversation context
            conversation_manager.add_user_message(conversation_id, user_query)
            
            # Check if this is a follow-up query
            is_follow_up = conversation_manager.is_follow_up_query(conversation_id, user_query)
            if is_follow_up:
                logger.info(f"Detected follow-up query for conversation {conversation_id}")
            
            # Step 1: Classify intent
            yield {
                "event": "thinking",
                "data": json.dumps({
                    "status": "Analyzing your request...",
                    "step": 1,
                    "total_steps": 5
                })
            }
            await asyncio.sleep(0.1)  # Small delay for SSE flush
            
            intent = llm_service.classify_query_intent(user_query)
            logger.info(f"Query intent: {intent}")
            
            # Get conversation context (needed for both paths)
            conversation_context = conversation_manager.get_context_for_llm(conversation_id)
            
            # =========== EXPLANATORY PATH ===========
            # For reasoning/advice/explanation questions, use LLM reasoning instead of SQL
            if intent.get("intent") == "EXPLANATORY":
                logger.info(f"Routing to EXPLANATORY path for: {user_query}")
                
                yield {
                    "event": "thinking",
                    "data": json.dumps({
                        "status": "Generating insights and recommendations...",
                        "step": 2,
                        "total_steps": 3
                    })
                }
                await asyncio.sleep(0.1)
                
                # Generate explanatory response from context
                explanatory_response = llm_service.generate_advisory_response(
                    user_query, 
                    conversation_context or "No previous data context available."
                )
                
                yield {
                    "event": "explanatory",
                    "data": json.dumps(explanatory_response)
                }
                
                # Track assistant response
                conversation_manager.add_assistant_message(
                    conversation_id,
                    explanatory_response.get("summary", ""),
                    sql_query=None,
                    data_summary=f"Explanatory: {len(explanatory_response.get('recommendations', []))} recommendations"
                )
                
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "status": "done",
                        "query_id": conversation_id,
                        "type": "explanatory"
                    })
                }
                return
            
            # =========== DATA QUERY PATH ===========
            # Step 2: Get schema and generate SQL (with conversation context)
            yield {
                "event": "thinking",
                "data": json.dumps({
                    "status": "Understanding your data needs...",
                    "step": 2,
                    "total_steps": 5
                })
            }
            await asyncio.sleep(0.1)
            
            schema_for_llm = schema_service.get_schema_for_llm()
            
            # Include conversation context for better follow-up handling
            if conversation_context:
                enhanced_query = f"{user_query}\n\n--- CONVERSATION CONTEXT ---\n{conversation_context}"
            else:
                enhanced_query = user_query
            
            sql_result = llm_service.generate_sql(enhanced_query, schema_for_llm)
            
            if not sql_result.get("sql"):
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "message": "I couldn't generate a query for your request. Could you please rephrase or be more specific?",
                        "details": sql_result.get("explanation", "Unknown error")
                    })
                }
                return
            
            yield {
                "event": "sql_generated",
                "data": json.dumps({
                    "sql": sql_result["sql"],
                    "explanation": sql_result.get("explanation", ""),
                    "tables_used": sql_result.get("tables_used", []),
                    "confidence": sql_result.get("confidence", "MEDIUM")
                })
            }
            await asyncio.sleep(0.1)
            
            # Step 3: Execute SQL query (with self-healing retry loop)
            yield {
                "event": "thinking",
                "data": json.dumps({
                    "status": "Retrieving data from database...",
                    "step": 3,
                    "total_steps": 5
                })
            }
            await asyncio.sleep(0.1)
            
            # Retry loop for SQL execution
            max_retries = 3
            current_sql = sql_result["sql"]
            query_result = None
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    query_result = sql_executor.execute(current_sql)
                    break  # Success - exit retry loop
                    
                except (SQLValidationError, SQLExecutionError) as e:
                    last_error = str(e)
                    logger.warning(f"SQL attempt {attempt + 1}/{max_retries} failed: {last_error}")
                    
                    # If we have retries left, try to fix the SQL
                    if attempt < max_retries - 1:
                        yield {
                            "event": "thinking",
                            "data": json.dumps({
                                "status": f"Query failed, analyzing and retrying... (attempt {attempt + 2}/{max_retries})",
                                "step": 3,
                                "total_steps": 5
                            })
                        }
                        await asyncio.sleep(0.1)
                        
                        # Use LLM to analyze and fix the SQL
                        fix_result = llm_service.analyze_and_fix_sql(
                            failed_sql=current_sql,
                            error_message=last_error,
                            original_query=user_query,
                            schema=schema_for_llm
                        )
                        
                        if fix_result.get("can_fix") and fix_result.get("fixed_sql"):
                            logger.info(f"SQL fix applied: {fix_result.get('fix_applied')}")
                            current_sql = fix_result["fixed_sql"]
                            
                            # Emit event to show the corrected SQL
                            yield {
                                "event": "sql_retry",
                                "data": json.dumps({
                                    "attempt": attempt + 2,
                                    "error_analysis": fix_result.get("error_analysis", ""),
                                    "fix_applied": fix_result.get("fix_applied", ""),
                                    "corrected_sql": current_sql
                                })
                            }
                            await asyncio.sleep(0.1)
                        else:
                            # LLM cannot fix - stop retrying
                            logger.warning(f"LLM cannot fix SQL: {fix_result.get('error_analysis')}")
                            break
            
            # Check if we got a successful result
            if query_result is None:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "message": "Query execution failed after all attempts",
                        "details": last_error,
                        "attempts": min(max_retries, attempt + 1)
                    })
                }
                return
            
            data = query_result["data"]
            row_count = query_result["row_count"]
            
            # Update conversation context with query result
            data_summary = f"{row_count} rows"
            if data:
                columns = list(data[0].keys())
                data_summary += f" with columns: {', '.join(columns[:5])}"
            
            conversation_manager.update_data_context(
                conversation_id,
                sql_result["sql"],
                data_summary,
                sql_result.get("tables_used", [])
            )
            
            yield {
                "event": "data_retrieved",
                "data": json.dumps({
                    "row_count": row_count,
                    "preview": data[:5] if data else [],
                    "columns": list(data[0].keys()) if data else [],
                    "limited": query_result.get("limited", False)
                })
            }
            await asyncio.sleep(0.1)
            
            # Step 4: Generate insights
            yield {
                "event": "thinking",
                "data": json.dumps({
                    "status": "Analyzing results and generating insights...",
                    "step": 4,
                    "total_steps": 5
                })
            }
            await asyncio.sleep(0.1)
            
            insights = llm_service.generate_insights(
                user_query, 
                sql_result["sql"], 
                data, 
                row_count
            )
            
            yield {
                "event": "analysis",
                "data": json.dumps({
                    "summary": insights.get("summary", ""),
                    "insights": insights.get("insights", []),
                    "recommendations": insights.get("recommendations", []),
                    "anomalies": insights.get("anomalies", []),
                    "trend": insights.get("trend")
                })
            }
            await asyncio.sleep(0.1)
            
            # Step 5: Generate visualization
            yield {
                "event": "thinking",
                "data": json.dumps({
                    "status": "Creating visualization...",
                    "step": 5,
                    "total_steps": 5
                })
            }
            await asyncio.sleep(0.1)
            
            if data:
                columns = list(data[0].keys())
                chart_config = llm_service.determine_chart_type(user_query, data, columns)
                full_chart_config = chart_generator.generate_from_llm_config(chart_config, data)
                metrics_cards = chart_generator.create_metrics_cards(insights)
                
                yield {
                    "event": "visualization",
                    "data": json.dumps({
                        "chartConfig": full_chart_config,
                        "metrics": metrics_cards,
                        "reasoning": chart_config.get("reasoning", "")
                    })
                }
            
            # Step 6: Generate follow-up suggestions
            await asyncio.sleep(0.1)
            
            # Create data summary for suggestions
            data_summary = f"{row_count} rows returned"
            if data:
                columns = list(data[0].keys())
                data_summary += f" with columns: {', '.join(columns[:5])}"
            
            suggestions = llm_service.generate_follow_up_suggestions(
                user_query,
                sql_result.get("sql", ""),
                data_summary,
                insights.get("insights", [])
            )
            
            yield {
                "event": "suggestions",
                "data": json.dumps(suggestions)
            }
            
            # Complete
            yield {
                "event": "complete",
                "data": json.dumps({
                    "status": "done",
                    "query_id": request.conversation_id
                })
            }
            
        except Exception as e:
            logger.error(f"Analytics chat error: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({
                    "message": "An unexpected error occurred",
                    "details": str(e)
                })
            }
    
    return EventSourceResponse(event_generator())


# Simple query endpoint (non-streaming, for testing)
@app.post("/api/analytics/query")
async def analytics_query(request: ChatRequest):
    """
    Non-streaming analytics query endpoint (for testing/simple integrations)
    """
    try:
        user_query = request.query
        
        # Get schema and generate SQL
        schema_for_llm = schema_service.get_schema_for_llm()
        sql_result = llm_service.generate_sql(user_query, schema_for_llm)
        
        if not sql_result.get("sql"):
            return {
                "success": False,
                "error": sql_result.get("explanation", "Failed to generate SQL")
            }
        
        # Execute query
        query_result = sql_executor.execute(sql_result["sql"])
        data = query_result["data"]
        
        # Generate insights
        insights = llm_service.generate_insights(
            user_query,
            sql_result["sql"],
            data,
            query_result["row_count"]
        )
        
        # Generate chart config
        if data:
            columns = list(data[0].keys())
            chart_config = llm_service.determine_chart_type(user_query, data, columns)
            full_chart_config = chart_generator.generate_from_llm_config(chart_config, data)
        else:
            full_chart_config = None
        
        return {
            "success": True,
            "sql": sql_result["sql"],
            "data": data,
            "row_count": query_result["row_count"],
            "insights": insights,
            "chart_config": full_chart_config
        }
        
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug
    )
