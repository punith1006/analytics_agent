"""
LLM Service - OpenAI Integration

Provides AI-powered capabilities for the analytics agent:
1. SQL query generation from natural language
2. Data analysis and insight generation
3. Chart type recommendation
4. Conversational responses

Uses OpenAI's GPT-4 with function calling for structured outputs.
"""
import json
import logging
from typing import Dict, Any, List, Optional, Generator
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    OpenAI LLM integration service for analytics agent
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o"  # Using GPT-4o for best performance
    
    def generate_sql(self, user_query: str, schema: str) -> Dict[str, Any]:
        """
        Generate a SQL query from natural language using the database schema
        
        Args:
            user_query: Natural language query from user
            schema: Database schema description
            
        Returns:
            Dictionary with generated SQL and explanation
        """
        system_prompt = f"""You are an expert SQL query generator for a Learning Management System (LMS) database.

DATABASE SCHEMA:
{schema}

RULES:
1. Generate ONLY valid MySQL SELECT queries
2. NEVER generate INSERT, UPDATE, DELETE, DROP, or any data-modifying statements
3. Use proper table aliases for clarity
4. Include appropriate JOINs when data from multiple tables is needed
5. Use DATE functions for time-based queries (DATE_SUB, DATE_FORMAT, etc.)
6. Always include a reasonable LIMIT clause (max 1000 rows)
7. Handle NULL values appropriately
8. Add comments explaining complex logic

IMPORTANT SEMANTIC MAPPINGS:
- "students", "users", "learners" -> usercredential table
- "courses", "programs" -> course table  
- "enrollments", "registrations" -> courseenrollment table
- "schedules" -> courseschedule table

Respond with a JSON object containing:
- sql: The generated SQL query
- explanation: Brief explanation of what the query does
- tables_used: List of tables referenced
- confidence: "HIGH", "MEDIUM", or "LOW" based on query complexity and ambiguity
"""

        user_prompt = f"""Generate a SQL query for the following request:

"{user_query}"

Respond ONLY with a valid JSON object."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistent SQL generation
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Generated SQL: {result.get('sql', '')[:100]}...")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {
                "sql": None,
                "explanation": f"Failed to parse LLM response: {str(e)}",
                "error": str(e),
                "confidence": "LOW"
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"LLM SQL generation error: {error_msg}")
            # Check for common OpenAI errors
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                explanation = "OpenAI API key is invalid or not configured. Please check your .env file."
            elif "rate_limit" in error_msg.lower():
                explanation = "Rate limit reached. Please wait a moment and try again."
            elif "quota" in error_msg.lower():
                explanation = "OpenAI API quota exceeded. Please check your billing."
            else:
                explanation = f"OpenAI API error: {error_msg}"
            return {
                "sql": None,
                "explanation": explanation,
                "error": error_msg,
                "confidence": "LOW"
            }
    
    def generate_insights(self, user_query: str, sql_query: str, 
                         data: List[Dict], row_count: int) -> Dict[str, Any]:
        """
        Generate business insights from query results
        
        Args:
            user_query: Original user query
            sql_query: The SQL query that was executed
            data: Query result data (first 50 rows for context)
            row_count: Total number of rows returned
            
        Returns:
            Dictionary with insights, summary, and recommendations
        """
        # Limit data sent to LLM to avoid token limits
        sample_data = data[:50] if len(data) > 50 else data
        
        system_prompt = """You are a data analyst expert for a Learning Management System.
Analyze the query results and provide actionable business insights.

Respond with a JSON object containing:
- summary: A 1-2 sentence summary of the key finding
- key_metrics: Array of {label, value, trend} objects for important metrics
- insights: Array of detailed insight strings (2-4 insights)
- anomalies: Array of any unusual patterns or outliers detected
- recommendations: Array of actionable recommendations (1-3 items)
- trend: Overall trend indicator (e.g., "+15%", "-5%", "stable") if applicable"""

        user_prompt = f"""User asked: "{user_query}"

SQL Query: {sql_query}

Query returned {row_count} rows. Sample data:
{json.dumps(sample_data, indent=2, default=str)}

Analyze this data and provide insights."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1500
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"LLM insight generation error: {e}")
            return {
                "summary": f"Query returned {row_count} rows.",
                "key_metrics": [],
                "insights": ["Unable to generate detailed insights."],
                "anomalies": [],
                "recommendations": [],
                "error": str(e)
            }
    
    def determine_chart_type(self, user_query: str, data: List[Dict], 
                            columns: List[str]) -> Dict[str, Any]:
        """
        Determine the best chart type for visualizing the data
        
        Args:
            user_query: Original user query
            data: Query result data (sample)
            columns: Column names in the result
            
        Returns:
            Chart configuration dictionary
        """
        sample_data = data[:10] if len(data) > 10 else data
        
        system_prompt = """You are a data visualization expert.
Determine the best chart type for the given data and query.

Available chart types:
- line: For time series and trends
- bar: For categorical comparisons
- area: For cumulative data over time
- pie: For part-to-whole relationships (use sparingly, max 6 segments)
- scatter: For correlation analysis

Respond with a JSON object containing:
- chartType: One of "line", "bar", "area", "pie", "scatter"
- title: Chart title
- xAxis: {dataKey: string, label: string} - the column for X axis
- yAxis: {dataKey: string, label: string} - the column for Y axis (or value)
- series: Array of {dataKey, name, color} for each data series
- reasoning: Brief explanation of why this chart type was chosen"""

        user_prompt = f"""User query: "{user_query}"

Columns available: {columns}

Sample data:
{json.dumps(sample_data, indent=2, default=str)}

Recommend the best visualization."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=800
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Chart type determination error: {e}")
            # Return default bar chart config
            return {
                "chartType": "bar",
                "title": "Query Results",
                "xAxis": {"dataKey": columns[0] if columns else "x", "label": columns[0] if columns else "X"},
                "yAxis": {"dataKey": columns[1] if len(columns) > 1 else "y", "label": columns[1] if len(columns) > 1 else "Y"},
                "series": [{"dataKey": columns[1] if len(columns) > 1 else columns[0], "name": "Value", "color": "#3b82f6"}],
                "error": str(e)
            }
    
    def stream_response(self, user_query: str, context: str) -> Generator[str, None, None]:
        """
        Stream a conversational response for general queries
        
        Args:
            user_query: User's question
            context: Additional context (previous messages, etc.)
            
        Yields:
            Response text chunks
        """
        system_prompt = """You are an intelligent LMS analytics assistant. 
You help administrators understand their data and make data-driven decisions.
Be concise, professional, and actionable in your responses.
If you need clarification, ask specific questions."""

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context}\n\nUser: {user_query}"}
                ],
                stream=True,
                temperature=0.7,
                max_tokens=500
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield f"I apologize, but I encountered an error: {str(e)}"
    
    def classify_query_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Classify the intent of a user query to determine how to handle it
        
        Args:
            user_query: User's input
            
        Returns:
            Intent classification with confidence
        """
        system_prompt = """Classify the user's query intent for an LMS analytics system.

Possible intents:
- DATA_QUERY: User wants to retrieve/analyze data (e.g., "show me enrollments")
- CHART_REQUEST: User explicitly wants a visualization (e.g., "graph of...")
- FORECAST: User wants predictions/projections (e.g., "predict next month...")
- CLARIFICATION: User is asking for help or clarification
- GENERAL: General question not requiring data access

Respond with JSON: {intent, confidence, reasoning}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=200
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return {
                "intent": "DATA_QUERY",
                "confidence": "LOW",
                "reasoning": "Fallback classification due to error"
            }
