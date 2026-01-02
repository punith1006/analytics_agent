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
9. CRITICAL: For string comparisons (titles, names, descriptions), ALWAYS use `LIKE` with wildcards (e.g. `LIKE '%Search Term%'`) instead of exact `=` match.
10. SEARCH STRATEGY: If the user search term seems like an abbreviation or partial name, use broad `LIKE` patterns.
    - Example: "prompt eng" -> `title LIKE '%Prompt%' AND title LIKE '%Eng%'`
    - Example: "ML" -> `title LIKE '%ML%'` OR `title LIKE '%Machine Learning%'`

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

CRITICAL: For dataKey values, you MUST use the EXACT column names from the data, NOT friendly names.
- If the data has column "average_rating", use "average_rating" as dataKey, NOT "Average Rating"
- The "name" field can be a friendly display name, but "dataKey" MUST MATCH the exact column name

Respond with a JSON object containing:
- chartType: One of "line", "bar", "area", "pie", "scatter"
- title: Chart title (friendly name)
- xAxis: {dataKey: string (EXACT column name), label: string (display name)}
- yAxis: {dataKey: string (EXACT column name), label: string (display name)}
- series: Array of {dataKey: EXACT column name, name: friendly display name, color: hex color}
- reasoning: Brief explanation of why this chart type was chosen"""

        user_prompt = f"""User query: "{user_query}"

Columns available (USE THESE EXACT NAMES for dataKey): {columns}

Sample data:
{json.dumps(sample_data, indent=2, default=str)}

Remember: dataKey must use EXACT column names like "{columns[0] if columns else 'column_name'}", not friendly names.

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
- OBJECTIVE: User wants to retrieve/analyze specific data from the database. Needs SQL to answer. (e.g., "show me enrollments", "list courses", "how many students")
- CHART_REQUEST: User explicitly wants a visualization (e.g., "graph of...", "chart showing...")
- EXPLANATORY: User wants reasoning, advice, explanations, or insights that DON'T require querying new data. Uses existing context + LLM reasoning. (e.g., "how to improve", "what should we do", "why did this happen")
- FORECAST: User wants predictions/projections (e.g., "predict next month...")
- CLARIFICATION: User is asking for help or clarification about how to use the system
- GENERAL: General greeting or off-topic question

IMPORTANT: 
- OBJECTIVE = needs to query database for facts
- EXPLANATORY = reasoning from existing context (advisory, diagnostic, why questions)

Examples:
- "Show me enrollment trends" → OBJECTIVE
- "How can we boost enrollments during low periods?" → EXPLANATORY
- "What strategies should we use to increase student retention?" → EXPLANATORY
- "List all courses" → OBJECTIVE
- "Why are enrollments dropping?" → EXPLANATORY (needs reasoning, not just data)
- "What does this trend tell us?" → EXPLANATORY

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
    
    def analyze_and_fix_sql(self, failed_sql: str, error_message: str, 
                            original_query: str, schema: str) -> Dict[str, Any]:
        """
        Analyze a failed SQL query and generate a corrected version.
        
        Args:
            failed_sql: The SQL query that failed
            error_message: The error message from the database
            original_query: The original user query
            schema: Database schema for context
            
        Returns:
            Dictionary with corrected SQL and explanation
        """
        system_prompt = """You are a SQL debugging expert. A SQL query has failed execution.
Your job is to:
1. Analyze the error message to understand WHY it failed
2. Look at the schema to find the CORRECT column/table names
3. Generate a FIXED SQL query that will work

COMMON ERRORS AND FIXES:
- "Unknown column 'x'": The column name is wrong. Find the correct column name in the schema.
- "Table doesn't exist": The table name is wrong. Use exact table names from schema.
- "Syntax error": Fix the SQL syntax.
- "Data truncated": Check data types and constraints.

CRITICAL RULES:
1. Use ONLY columns that exist in the schema
2. Use EXACT column names as shown in the schema (case-sensitive)
3. Do NOT invent columns - if you can't find a matching column, explain why

Respond with JSON:
{
    "fixed_sql": "The corrected SQL query",
    "error_analysis": "Brief explanation of what was wrong",
    "fix_applied": "What was changed to fix it",
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "can_fix": true | false
}

If you cannot fix the query (e.g., required data doesn't exist), set can_fix to false and explain why."""

        user_prompt = f"""FAILED SQL:
```sql
{failed_sql}
```

ERROR MESSAGE:
{error_message}

ORIGINAL USER QUERY:
{original_query}

DATABASE SCHEMA:
{schema}

Analyze the error and provide a corrected SQL query."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for precise fixes
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"SQL fix analysis: {result.get('error_analysis', 'No analysis')}")
            return result
            
        except Exception as e:
            logger.error(f"SQL fix analysis error: {e}")
            return {
                "fixed_sql": None,
                "error_analysis": f"Could not analyze error: {str(e)}",
                "fix_applied": None,
                "confidence": "LOW",
                "can_fix": False
            }
    
    def generate_advisory_response(self, user_query: str, conversation_context: str) -> Dict[str, Any]:
        """
        Generate strategic advice and recommendations based on conversation context.
        Used for ADVISORY intent queries that don't need new SQL queries.
        
        Args:
            user_query: The advisory question from the user
            conversation_context: Previous conversation including data results
            
        Returns:
            Dictionary with advice, recommendations, and optional action items
        """
        system_prompt = """You are a strategic analytics advisor for a Learning Management System.
The user is asking for advice or recommendations based on data they've already seen in this conversation.

Your job is to:
1. Analyze the question in context of the available data
2. Provide strategic, actionable recommendations
3. Be specific - reference actual numbers/trends from the context
4. Structure your response clearly

Response format (JSON):
{
    "summary": "Brief 1-2 sentence executive summary of your advice",
    "key_insight": "The most important finding that drives your recommendation",
    "recommendations": [
        {
            "title": "Short action title",
            "description": "Detailed explanation of what to do and why",
            "priority": "HIGH" | "MEDIUM" | "LOW",
            "expected_impact": "What improvement to expect"
        }
    ],
    "reasoning": "Brief explanation of how you arrived at these recommendations based on the data"
}

Be practical and specific. Avoid generic advice. Reference actual data from the conversation."""

        user_prompt = f"""Question: {user_query}

CONVERSATION CONTEXT (including previous data and insights):
{conversation_context}

Based on the above context, provide strategic recommendations."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,  # Slightly higher for creative recommendations
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Generated advisory response with {len(result.get('recommendations', []))} recommendations")
            return result
            
        except Exception as e:
            logger.error(f"Advisory response generation error: {e}")
            return {
                "summary": "I apologize, but I couldn't generate recommendations at this time.",
                "key_insight": "Please try rephrasing your question.",
                "recommendations": [],
                "reasoning": f"Error: {str(e)}"
            }
    
    def generate_follow_up_suggestions(self, user_query: str, sql_query: str, 
                                       data_summary: str, insights: List[str],
                                       columns: List[str] = None,
                                       row_count: int = 0,
                                       data_time_range: str = None) -> Dict[str, Any]:
        """
        Generate context-aware follow-up question suggestions based on the current query and results.
        
        Args:
            user_query: Original user query
            sql_query: The SQL query that was executed
            data_summary: Summary of the data returned
            insights: Key insights generated
            columns: Available columns in the result (for schema awareness)
            row_count: Number of rows returned (for conditional display)
            data_time_range: Time range of data if applicable (e.g., "Jan 2024 - Dec 2024")
            
        Returns:
            Dictionary with categorized follow-up suggestions
        """
        # Skip suggestions for single-value or empty results
        if row_count <= 1:
            logger.info(f"Skipping suggestions for simple result (row_count={row_count})")
            return {"categories": [], "skip_reason": "single_value_result"}
        
        # Build context for schema-aware suggestions
        columns_info = f"Available columns: {', '.join(columns)}" if columns else ""
        time_range_info = f"Data time range: {data_time_range}" if data_time_range else "Time range: unknown"
        
        system_prompt = """You are an expert data analyst assistant for an LMS (Learning Management System).
Based on the user's query and the results, suggest follow-up questions they might want to ask.

Generate suggestions in THREE categories:
1. DRILL_DEEPER: Questions that explore the data in more detail (type: "objective")
2. COMPARE: Questions that compare with other time periods, segments, or benchmarks (type: "objective")
3. ACTION: Questions about strategy, improvements, or "how to" advice (type: "explanatory")

CRITICAL RULES:
- Each category should have 1-2 suggestions maximum
- Suggestions should be SHORT (under 8 words for the label)
- Suggestions MUST be specific to the current data context
- DO NOT suggest generic questions like "Show by category" if the data doesn't have categories
- DO NOT suggest time comparisons if the data time range doesn't support it
- Only suggest drilling into dimensions that exist in the available columns
- If you can't generate relevant suggestions for a category, leave it empty

QUALITY CHECKS:
- "Compare with last year" → Only if data range spans multiple years
- "Break down by X" → Only if column X exists in the data
- "How to improve" → Always valid as an explanatory question

Respond with JSON:
{
  "categories": [
    {
      "name": "Drill Deeper",
      "icon": "📊",
      "suggestions": [
        {"text": "Short label", "query": "Full natural language query", "type": "objective"}
      ]
    },
    {
      "name": "Compare",
      "icon": "🔍",
      "suggestions": [
        {"text": "vs last month", "query": "Compare with last month", "type": "objective"}
      ]
    },
    {
      "name": "Take Action", 
      "icon": "🎯",
      "suggestions": [
        {"text": "How to improve", "query": "How can we improve these numbers?", "type": "explanatory"}
      ]
    }
  ]
}

If a category has no relevant suggestions, return it with an empty suggestions array."""

        user_prompt = f"""User asked: "{user_query}"

SQL Query executed:
{sql_query}

Data Summary: {data_summary} ({row_count} rows)
{columns_info}
{time_range_info}

Key Insights: {', '.join(insights[:3]) if insights else 'No specific insights'}

Generate relevant, specific follow-up suggestions. Do NOT include generic suggestions that don't fit the data."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5,  # Balanced for relevance
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Filter out empty categories
            if "categories" in result:
                result["categories"] = [
                    cat for cat in result["categories"] 
                    if cat.get("suggestions") and len(cat["suggestions"]) > 0
                ]
            
            logger.info(f"Generated {len(result.get('categories', []))} suggestion categories")
            return result
            
        except Exception as e:
            logger.error(f"Follow-up suggestion generation error: {e}")
            # Return empty instead of generic suggestions
            return {
                "categories": [],
                "error": str(e)
            }

    def detect_query_ambiguity(self, user_query: str, schema_context: str) -> Dict[str, Any]:
        """
        Detect if a user query is ambiguous and needs clarification.
        
        Args:
            user_query: The user's natural language query
            schema_context: Available database schema for context
            
        Returns:
            Dictionary with ambiguity detection results and clarification options
        """
        system_prompt = """You are an expert at understanding user intent for LMS analytics queries.

Analyze if the user's query is AMBIGUOUS and needs clarification before generating SQL.

A query is ambiguous if:
1. TIME RANGE is unclear (e.g., "recent", "top", without specifying period)
2. METRIC is unclear (e.g., "performance" could mean completion rate, grades, or engagement)
3. SCOPE is unclear (e.g., "students" could mean all, active, or specific cohort)
4. COMPARISON is implied but not specified (e.g., "how are we doing" - compared to what?)

If the query IS clear enough to proceed, return:
{"needs_clarification": false, "confidence": "HIGH", "reasoning": "..."}

If clarification needed, return:
{
  "needs_clarification": true,
  "confidence": "MEDIUM" or "LOW",
  "reasoning": "Brief explanation",
  "clarification": {
    "question": "The question to ask user",
    "type": "time_range" | "metric" | "scope" | "comparison",
    "options": [
      {"id": "option1", "icon": "📅", "label": "This month"},
      {"id": "option2", "icon": "📆", "label": "This quarter"},
      ...
    ],
    "default": "option1"
  }
}

Be practical - don't ask for clarification if reasonable defaults exist."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: \"{user_query}\"\n\nAvailable tables: {schema_context[:1000]}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=400
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Ambiguity check: needs_clarification={result.get('needs_clarification', False)}")
            return result
            
        except Exception as e:
            logger.error(f"Ambiguity detection error: {e}")
            return {
                "needs_clarification": False,
                "confidence": "MEDIUM",
                "reasoning": "Proceeding with defaults due to detection error"
            }

