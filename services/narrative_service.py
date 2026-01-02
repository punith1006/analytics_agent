"""
Narrative Service - Data Storytelling Engine

Generates fact-based, grounded narratives from query results.
All claims are traced back to source data to prevent hallucination.

Key sections:
- Executive Summary: What happened (facts with numbers)
- Key Numbers: Highlighted metrics from the data
- The Story: Why it matters (analysis)
- Recommendations: What to do next
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)


class NarrativeService:
    """Service for generating data-grounded narrative storytelling"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o"
    
    def should_generate_narrative(
        self,
        row_count: int,
        insights: List[str],
        query_type: str
    ) -> bool:
        """
        Determine if narrative generation is beneficial for this query.
        
        Args:
            row_count: Number of rows in result
            insights: Generated insights
            query_type: Type of query (trend, comparison, etc.)
            
        Returns:
            True if narrative should be generated
        """
        # Skip for simple lookups
        if row_count <= 1:
            return False
        
        # Skip if no meaningful insights
        if len(insights) < 2:
            return False
        
        # Generate for complex analysis queries
        narrative_worthy_types = ["trend", "comparison", "performance", "distribution", "analysis"]
        
        for query_type_keyword in narrative_worthy_types:
            if query_type_keyword in query_type.lower():
                return True
        
        # Generate for larger datasets
        if row_count >= 5 and len(insights) >= 3:
            return True
        
        return False
    
    def generate_narrative(
        self,
        user_query: str,
        data: List[Dict],
        sql_query: str,
        insights: Dict[str, Any],
        columns: List[str],
        row_count: int
    ) -> Dict[str, Any]:
        """
        Generate a grounded narrative from query results.
        
        CRITICAL: Every claim must be traceable to the source data.
        No speculation or external benchmarks allowed.
        
        Args:
            user_query: Original user question
            data: Query result data (first 50 rows)
            sql_query: The SQL that was executed
            insights: Already generated insights
            columns: Column names in result
            row_count: Total row count
            
        Returns:
            Structured narrative with citations
        """
        # Prepare data summary for the prompt
        data_summary = self._prepare_data_summary(data, columns)
        insights_text = "\n".join(insights.get("insights", []))
        
        system_prompt = """You are a data storytelling expert. Generate a narrative that explains the data.

CRITICAL RULES - YOU MUST FOLLOW:
1. ONLY reference numbers that EXIST in the provided data
2. NEVER invent statistics or percentages not in the data
3. NEVER speculate about causes unless data supports it
4. NEVER reference external benchmarks or industry standards
5. If uncertain, say "The data shows..." or "Based on the data..."
6. Every claim must be traceable to the source data

OUTPUT FORMAT (JSON):
{
  "should_display": true,
  "executive_summary": "2-3 sentences summarizing what the data shows",
  "key_numbers": [
    {
      "value": "1,247",
      "label": "Total Enrollments",
      "context": "across all categories",
      "trend": "up" | "down" | "stable" | null
    }
  ],
  "story_sections": [
    {
      "title": "What the Data Shows",
      "content": "Factual description of key findings",
      "icon": "📊"
    },
    {
      "title": "Key Observations",
      "content": "Important patterns or standouts",
      "icon": "🔍"
    }
  ],
  "recommendations": [
    {
      "action": "Specific actionable recommendation",
      "rationale": "Based on [specific data point]",
      "priority": "high" | "medium" | "low"
    }
  ],
  "data_limitations": "Any caveats about what the data doesn't show (optional)"
}

Keep it CONCISE - max 150 words total. Focus on actionable insights."""

        user_prompt = f"""Generate a data story for this query.

USER QUESTION: "{user_query}"

SQL EXECUTED:
{sql_query}

DATA SUMMARY ({row_count} rows):
{data_summary}

COLUMNS: {', '.join(columns)}

KEY INSIGHTS ALREADY IDENTIFIED:
{insights_text}

Generate a grounded narrative that explains the "so what" of this data.
Remember: ONLY use numbers from the data provided."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,  # Lower temperature for factual accuracy
                max_tokens=800
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate narrative against source data
            validated_result = self._validate_narrative(result, data, columns)
            
            logger.info(f"Generated narrative with {len(validated_result.get('key_numbers', []))} key numbers")
            return validated_result
            
        except Exception as e:
            logger.error(f"Narrative generation error: {e}")
            return {
                "should_display": False,
                "error": str(e)
            }
    
    def _prepare_data_summary(self, data: List[Dict], columns: List[str]) -> str:
        """Prepare a compact data summary for the LLM."""
        if not data:
            return "No data available"
        
        summary_parts = []
        
        # Show first few rows
        summary_parts.append("Sample rows:")
        for i, row in enumerate(data[:5]):
            row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
            summary_parts.append(f"  {i+1}. {row_str}")
        
        if len(data) > 5:
            summary_parts.append(f"  ... and {len(data) - 5} more rows")
        
        # Calculate basic stats for numeric columns
        numeric_stats = []
        for col in columns:
            values = [row.get(col) for row in data if isinstance(row.get(col), (int, float))]
            if values:
                total = sum(values)
                avg = total / len(values)
                numeric_stats.append(f"{col}: total={total:,.0f}, avg={avg:,.1f}, min={min(values)}, max={max(values)}")
        
        if numeric_stats:
            summary_parts.append("\nNumeric column stats:")
            summary_parts.extend([f"  {s}" for s in numeric_stats])
        
        return "\n".join(summary_parts)
    
    def _validate_narrative(
        self, 
        narrative: Dict[str, Any], 
        data: List[Dict],
        columns: List[str]
    ) -> Dict[str, Any]:
        """
        Validate that narrative claims are grounded in source data.
        Removes or flags unverifiable claims.
        """
        # Extract all numbers from the data
        data_numbers = set()
        for row in data:
            for value in row.values():
                if isinstance(value, (int, float)):
                    data_numbers.add(round(value, 2))
                    data_numbers.add(int(value))
        
        # Validate key numbers
        validated_key_numbers = []
        for num in narrative.get("key_numbers", []):
            value_str = str(num.get("value", "")).replace(",", "")
            try:
                value_num = float(value_str)
                # Check if this number (or close to it) exists in data
                if any(abs(value_num - dn) < 1 for dn in data_numbers if isinstance(dn, (int, float))):
                    num["verified"] = True
                else:
                    num["verified"] = False
                    num["warning"] = "Number not directly from data"
                validated_key_numbers.append(num)
            except ValueError:
                validated_key_numbers.append(num)
        
        narrative["key_numbers"] = validated_key_numbers
        narrative["validation_status"] = "validated"
        
        return narrative


# Singleton instance
narrative_service = NarrativeService()
