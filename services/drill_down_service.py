"""
Drill-Down Service - Interactive Data Exploration

Enables users to click on chart elements and drill into contextual breakdowns.
Generates SQL queries to explore data hierarchically:
Category → Course → Module → Student
"""
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config import settings
import json

logger = logging.getLogger(__name__)


class DrillDownService:
    """Service for generating drill-down queries from chart interactions"""
    
    # Define drill hierarchies for different dimensions
    DRILL_HIERARCHIES = {
        # Category-based drilling
        "category": ["category_name", "course_name", "module_name", "student_name"],
        "category_name": ["course_name", "module_name", "student_name"],
        
        # Partner-based drilling
        "partner": ["partner_name", "course_name", "enrollment_count"],
        "partner_name": ["course_name", "enrollment_count"],
        
        # Time-based drilling
        "year": ["quarter", "month", "date"],
        "quarter": ["month", "week", "date"],
        "month": ["week", "date"],
        
        # Course-based drilling
        "course": ["course_name", "module_name", "enrollment_date"],
        "course_name": ["module_name", "student_name", "enrollment_date"],
        
        # Default fallback
        "default": []
    }
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o"
    
    def get_drill_options(
        self, 
        clicked_element: Dict[str, Any],
        current_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate drill-down options for a clicked chart element.
        
        Args:
            clicked_element: {dimension, value, label, rawData}
            current_context: {sql_query, columns, tables_used}
            
        Returns:
            List of drill options with queries
        """
        dimension = clicked_element.get("dimension", "").lower()
        value = clicked_element.get("value")
        label = clicked_element.get("label", str(value))
        
        # Get available drill paths for this dimension
        drill_path = self.DRILL_HIERARCHIES.get(
            dimension, 
            self.DRILL_HIERARCHIES["default"]
        )
        
        options = []
        
        # Option 1: Breakdown by next level in hierarchy
        if drill_path:
            next_dimension = drill_path[0] if drill_path else None
            if next_dimension:
                options.append({
                    "id": "breakdown",
                    "icon": "📊",
                    "label": f"Break down by {next_dimension.replace('_', ' ')}",
                    "description": f"See {label}'s composition",
                    "drill_type": "breakdown",
                    "target_dimension": next_dimension
                })
        
        # Option 2: Show trend over time
        options.append({
            "id": "trend",
            "icon": "📈",
            "label": f"Show {label} trend over time",
            "description": "See how this has changed",
            "drill_type": "trend",
            "target_dimension": "time"
        })
        
        # Option 3: Compare with others
        options.append({
            "id": "compare",
            "icon": "🔍",
            "label": f"Compare {label} with others",
            "description": "See how it ranks",
            "drill_type": "compare",
            "target_dimension": dimension
        })
        
        # Option 4: Show details/list
        options.append({
            "id": "details",
            "icon": "📋",
            "label": f"Show all {label} records",
            "description": "View the underlying data",
            "drill_type": "details",
            "target_dimension": None
        })
        
        return options
    
    def generate_drill_query(
        self,
        clicked_element: Dict[str, Any],
        drill_option: Dict[str, Any],
        current_context: Dict[str, Any],
        schema: str
    ) -> Dict[str, Any]:
        """
        Generate a SQL query for the selected drill-down option.
        
        Args:
            clicked_element: The chart element that was clicked
            drill_option: The drill option selected by user
            current_context: Current query context (SQL, tables, etc.)
            schema: Database schema
            
        Returns:
            Dictionary with new SQL query and explanation
        """
        dimension = clicked_element.get("dimension", "")
        value = clicked_element.get("value")
        label = clicked_element.get("label", str(value))
        drill_type = drill_option.get("drill_type", "breakdown")
        target_dimension = drill_option.get("target_dimension")
        
        current_sql = current_context.get("sql_query", "")
        tables_used = current_context.get("tables_used", [])
        
        system_prompt = """You are an expert SQL analyst. Generate a drill-down query based on the user's interaction.

IMPORTANT RULES:
1. The new query MUST filter by the clicked element's value
2. Maintain consistency with the original query structure
3. Return results that make sense for the drill-down type
4. Keep the query efficient and readable

DRILL-DOWN TYPES:
- breakdown: Group by the next hierarchy level, filtered by clicked value
- trend: Show the clicked value over time periods
- compare: Show the clicked value alongside its peers for comparison
- details: Show individual records for the clicked value

Return JSON:
{
  "sql": "SELECT ...",
  "explanation": "This query shows...",
  "expected_chart": "bar|line|table",
  "title": "Chart title for this drill-down"
}"""

        user_prompt = f"""User clicked on a chart element:
- Dimension: {dimension}
- Value/Label: {label}
- Raw value: {value}

Drill-down requested: {drill_type}
Target dimension: {target_dimension or 'N/A'}

Original query context:
{current_sql}

Tables available: {', '.join(tables_used) if tables_used else 'See schema'}

Database Schema:
{schema[:2000]}

Generate the appropriate drill-down SQL query."""

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
            
            # Add metadata
            result["drill_context"] = {
                "parent_dimension": dimension,
                "parent_value": label,
                "drill_type": drill_type
            }
            
            logger.info(f"Generated drill-down query for {dimension}={label}, type={drill_type}")
            return result
            
        except Exception as e:
            logger.error(f"Drill-down query generation error: {e}")
            return {
                "sql": None,
                "error": str(e),
                "explanation": "Failed to generate drill-down query"
            }
    
    def build_breadcrumb(
        self,
        current_path: List[Dict[str, str]],
        new_element: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Build navigation breadcrumb for drill history.
        
        Args:
            current_path: Existing breadcrumb path
            new_element: New drill-down element to add
            
        Returns:
            Updated breadcrumb path
        """
        new_crumb = {
            "dimension": new_element.get("dimension", ""),
            "value": str(new_element.get("label", new_element.get("value", ""))),
            "drill_type": new_element.get("drill_type", "breakdown")
        }
        
        return current_path + [new_crumb]


# Singleton instance
drill_down_service = DrillDownService()
