"""
Schema Discovery Service

Discovers and caches database schema information for the LLM to use
when generating SQL queries. Queries information_schema to get
table structures, column types, and relationships.
"""
from typing import Dict, List, Any, Optional
from database import execute_raw_query
from config import settings
import json


class SchemaService:
    """Service for discovering and caching database schema"""
    
    def __init__(self):
        self._schema_cache: Optional[Dict[str, Any]] = None
        self._semantic_mappings: Dict[str, str] = {
            # Common natural language to table/column mappings
            "students": "usercredential",
            "users": "usercredential",
            "learners": "usercredential",
            "courses": "course",
            "programs": "course",
            "enrollments": "courseenrollment",
            "registrations": "courseenrollment",
            "schedules": "courseschedule",
            "categories": "coursecategory",
            "partners": "partner",
            "instructors": "courseinstructor",
            "certificates": "certificate",
            "webinars": "webinar",
            "offers": "offer",
        }
    
    def get_schema(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get complete database schema with caching
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh schema
            
        Returns:
            Dictionary containing tables, columns, and relationships
        """
        if self._schema_cache is None or force_refresh:
            self._schema_cache = self._discover_schema()
        return self._schema_cache
    
    def _discover_schema(self) -> Dict[str, Any]:
        """Discover schema from information_schema"""
        tables = self._get_tables()
        schema = {
            "database": settings.db_name,
            "tables": {},
            "relationships": [],
            "semantic_mappings": self._semantic_mappings
        }
        
        for table_name in tables:
            columns = self._get_columns(table_name)
            primary_key = self._get_primary_key(table_name)
            foreign_keys = self._get_foreign_keys(table_name)
            
            schema["tables"][table_name] = {
                "columns": columns,
                "primary_key": primary_key,
                "foreign_keys": foreign_keys,
                "row_count": self._get_row_count(table_name)
            }
            
            # Add relationships
            for fk in foreign_keys:
                schema["relationships"].append({
                    "from_table": table_name,
                    "from_column": fk["column"],
                    "to_table": fk["referenced_table"],
                    "to_column": fk["referenced_column"]
                })
        
        return schema
    
    def _get_tables(self) -> List[str]:
        """Get all table names in the database"""
        query = """
            SELECT TABLE_NAME 
            FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = :db_name
            AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """
        results = execute_raw_query(query, {"db_name": settings.db_name})
        return [row["TABLE_NAME"] for row in results]
    
    def _get_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Get column information for a table"""
        query = """
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_KEY,
                COLUMN_DEFAULT,
                COLUMN_COMMENT
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = :db_name
            AND TABLE_NAME = :table_name
            ORDER BY ORDINAL_POSITION
        """
        results = execute_raw_query(query, {
            "db_name": settings.db_name,
            "table_name": table_name
        })
        return [
            {
                "name": row["COLUMN_NAME"],
                "type": row["DATA_TYPE"],
                "nullable": row["IS_NULLABLE"] == "YES",
                "key": row["COLUMN_KEY"],
                "default": row["COLUMN_DEFAULT"],
                "comment": row["COLUMN_COMMENT"]
            }
            for row in results
        ]
    
    def _get_primary_key(self, table_name: str) -> Optional[str]:
        """Get primary key column for a table"""
        query = """
            SELECT COLUMN_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = :db_name
            AND TABLE_NAME = :table_name
            AND COLUMN_KEY = 'PRI'
        """
        results = execute_raw_query(query, {
            "db_name": settings.db_name,
            "table_name": table_name
        })
        return results[0]["COLUMN_NAME"] if results else None
    
    def _get_foreign_keys(self, table_name: str) -> List[Dict[str, str]]:
        """Get foreign key relationships for a table"""
        query = """
            SELECT 
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = :db_name
            AND TABLE_NAME = :table_name
            AND REFERENCED_TABLE_NAME IS NOT NULL
        """
        results = execute_raw_query(query, {
            "db_name": settings.db_name,
            "table_name": table_name
        })
        return [
            {
                "column": row["COLUMN_NAME"],
                "referenced_table": row["REFERENCED_TABLE_NAME"],
                "referenced_column": row["REFERENCED_COLUMN_NAME"]
            }
            for row in results
        ]
    
    def _get_row_count(self, table_name: str) -> int:
        """Get approximate row count for a table"""
        try:
            query = f"SELECT COUNT(*) as count FROM `{table_name}` LIMIT 1"
            results = execute_raw_query(query)
            return results[0]["count"] if results else 0
        except Exception:
            return 0
    
    def get_schema_for_llm(self) -> str:
        """
        Get a compact, LLM-friendly representation of the schema
        
        Returns:
            String description of schema suitable for LLM context
        """
        schema = self.get_schema()
        lines = [
            f"Database: {schema['database']}",
            f"Total Tables: {len(schema['tables'])}",
            "",
            "=== TABLE SCHEMAS ===",
            ""
        ]
        
        for table_name, table_info in schema["tables"].items():
            columns_str = ", ".join([
                f"{col['name']} ({col['type']}{'*' if col['key'] == 'PRI' else ''})"
                for col in table_info["columns"]
            ])
            lines.append(f"TABLE `{table_name}` ({table_info['row_count']} rows):")
            lines.append(f"  Columns: {columns_str}")
            
            if table_info["foreign_keys"]:
                fk_str = ", ".join([
                    f"{fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}"
                    for fk in table_info["foreign_keys"]
                ])
                lines.append(f"  Foreign Keys: {fk_str}")
            lines.append("")
        
        lines.append("=== SEMANTIC MAPPINGS ===")
        for term, table in schema["semantic_mappings"].items():
            lines.append(f"  '{term}' refers to table `{table}`")
        
        return "\n".join(lines)
    
    def resolve_table_name(self, natural_term: str) -> Optional[str]:
        """
        Resolve a natural language term to an actual table name
        
        Args:
            natural_term: Natural language reference (e.g., "students")
            
        Returns:
            Actual table name or None if not found
        """
        term_lower = natural_term.lower()
        
        # Check semantic mappings first
        if term_lower in self._semantic_mappings:
            return self._semantic_mappings[term_lower]
        
        # Check if it's an exact table name
        schema = self.get_schema()
        if term_lower in [t.lower() for t in schema["tables"].keys()]:
            for table_name in schema["tables"].keys():
                if table_name.lower() == term_lower:
                    return table_name
        
        return None
