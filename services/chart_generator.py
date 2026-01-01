"""
Chart Generator Service

Generates Recharts-compatible chart configurations from data and LLM recommendations.
Handles color schemes, formatting, and responsive configurations.
"""
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


# Professional color palette for charts
CHART_COLORS = [
    "#3b82f6",  # Blue
    "#8b5cf6",  # Purple
    "#ec4899",  # Pink
    "#f59e0b",  # Amber
    "#10b981",  # Emerald
    "#06b6d4",  # Cyan
    "#f97316",  # Orange
    "#84cc16",  # Lime
]

CHART_COLORS_LIGHT = [
    "#93c5fd",  # Light Blue
    "#c4b5fd",  # Light Purple
    "#f9a8d4",  # Light Pink
    "#fcd34d",  # Light Amber
    "#6ee7b7",  # Light Emerald
    "#67e8f9",  # Light Cyan
]


class ChartGenerator:
    """
    Generates Recharts-compatible chart configurations
    """
    
    def __init__(self):
        self.colors = CHART_COLORS
        self.colors_light = CHART_COLORS_LIGHT
    
    def generate_config(self, chart_type: str, data: List[Dict], 
                       x_axis_key: str, y_axis_key: str,
                       title: str = "Chart",
                       series_config: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate a complete Recharts configuration
        
        Args:
            chart_type: Type of chart (line, bar, area, pie, scatter)
            data: Data to visualize
            x_axis_key: Data key for X axis
            y_axis_key: Data key for Y axis
            title: Chart title
            series_config: Optional series configuration
            
        Returns:
            Recharts-compatible configuration dictionary
        """
        # Validate chart type
        valid_types = ['line', 'bar', 'area', 'pie', 'scatter']
        if chart_type not in valid_types:
            chart_type = 'bar'
        
        # Build base configuration
        config = {
            "type": chart_type,
            "title": title,
            "data": data,
            "xAxis": {
                "dataKey": x_axis_key,
                "label": self._format_label(x_axis_key),
                "angle": -45 if len(data) > 10 else 0,
                "tickMargin": 10
            },
            "yAxis": {
                "dataKey": y_axis_key,
                "label": self._format_label(y_axis_key),
                "tickFormatter": "number"
            },
            "tooltip": True,
            "legend": True,
            "responsive": True,
            "height": 400
        }
        
        # Add series configuration
        if series_config:
            config["series"] = series_config
        else:
            # Auto-generate series from data
            config["series"] = self._auto_generate_series(data, y_axis_key)
        
        # Chart-specific configurations
        if chart_type == 'line':
            config["lineType"] = "monotone"
            config["dot"] = True
            config["activeDot"] = {"r": 6}
        elif chart_type == 'bar':
            config["barSize"] = 40
            config["barGap"] = 4
        elif chart_type == 'area':
            config["fillOpacity"] = 0.3
            config["strokeWidth"] = 2
        elif chart_type == 'pie':
            config["innerRadius"] = 60  # Makes it a donut chart
            config["outerRadius"] = 120
            config["paddingAngle"] = 2
            config["dataKey"] = y_axis_key
            config["nameKey"] = x_axis_key
        elif chart_type == 'scatter':
            config["dotSize"] = 60
        
        return config
    
    def _format_label(self, key: str) -> str:
        """Convert snake_case or camelCase to Title Case"""
        # Handle snake_case
        if '_' in key:
            return ' '.join(word.capitalize() for word in key.split('_'))
        # Handle camelCase
        import re
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', key)
        return words.title()
    
    def _auto_generate_series(self, data: List[Dict], primary_key: str) -> List[Dict]:
        """Auto-generate series configuration from data structure"""
        if not data:
            return []
        
        # Get all numeric columns except the primary key
        sample = data[0]
        series = []
        color_index = 0
        
        for key, value in sample.items():
            if key == primary_key or isinstance(value, (int, float)):
                if key != primary_key or len(series) == 0:
                    series.append({
                        "dataKey": key if key != primary_key else primary_key,
                        "name": self._format_label(key if key != primary_key else primary_key),
                        "color": self.colors[color_index % len(self.colors)],
                        "fill": self.colors[color_index % len(self.colors)]
                    })
                    color_index += 1
        
        return series if series else [{"dataKey": primary_key, "name": "Value", "color": self.colors[0]}]
    
    def generate_from_llm_config(self, llm_config: Dict[str, Any], 
                                  data: List[Dict]) -> Dict[str, Any]:
        """
        Generate chart config from LLM recommendation
        
        Args:
            llm_config: Configuration from LLM service
            data: Query result data
            
        Returns:
            Complete Recharts configuration
        """
        chart_type = llm_config.get("chartType", "bar")
        title = llm_config.get("title", "Query Results")
        
        x_axis = llm_config.get("xAxis", {})
        y_axis = llm_config.get("yAxis", {})
        
        # Get actual column names from data
        actual_columns = list(data[0].keys()) if data else []
        
        x_key = x_axis.get("dataKey", actual_columns[0] if actual_columns else "x")
        y_key = y_axis.get("dataKey", actual_columns[1] if len(actual_columns) > 1 else "y")
        
        # Validate x_key exists in data
        if x_key not in actual_columns and actual_columns:
            logger.warning(f"xAxis dataKey '{x_key}' not found in data columns {actual_columns}, using first column")
            x_key = actual_columns[0]
        
        # Process series from LLM
        series_config = []
        for i, series in enumerate(llm_config.get("series", [])):
            data_key = series.get("dataKey", y_key)
            
            # Validate dataKey exists in actual data
            if data_key not in actual_columns:
                logger.warning(f"Series dataKey '{data_key}' not found in data columns {actual_columns}, skipping")
                continue
                
            series_config.append({
                "dataKey": data_key,
                "name": series.get("name", self._format_label(data_key)),
                "color": series.get("color", self.colors[i % len(self.colors)]),
                "fill": series.get("color", self.colors[i % len(self.colors)]),
                "strokeDasharray": series.get("strokeDasharray")  # For forecast lines
            })
        
        # If no valid series, auto-generate from numeric columns
        if not series_config:
            logger.info("No valid series from LLM config, auto-generating from data")
            series_config = self._auto_generate_series(data, x_key)
        
        return self.generate_config(
            chart_type=chart_type,
            data=data,
            x_axis_key=x_key,
            y_axis_key=y_key if y_key in actual_columns else (series_config[0]["dataKey"] if series_config else "y"),
            title=title,
            series_config=series_config if series_config else None
        )
    
    def create_metrics_cards(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create metric card configurations from insights
        
        Args:
            insights: Insights dictionary from LLM service
            
        Returns:
            List of metric card configurations
        """
        cards = []
        
        for i, metric in enumerate(insights.get("key_metrics", [])):
            cards.append({
                "label": metric.get("label", f"Metric {i+1}"),
                "value": metric.get("value", "N/A"),
                "trend": metric.get("trend"),
                "trendDirection": self._get_trend_direction(metric.get("trend", "")),
                "color": self.colors[i % len(self.colors)]
            })
        
        return cards
    
    def _get_trend_direction(self, trend: str) -> str:
        """Determine trend direction from trend string"""
        if not trend:
            return "neutral"
        if trend.startswith('+') or 'increase' in trend.lower() or 'up' in trend.lower():
            return "up"
        if trend.startswith('-') or 'decrease' in trend.lower() or 'down' in trend.lower():
            return "down"
        return "neutral"
