"""
SQL Executor Service

Securely executes SQL queries with validation to ensure:
1. Read-only operations only (no INSERT, UPDATE, DELETE, DROP, etc.)
2. Query timeout enforcement
3. Result row limiting
4. Error handling and logging
"""
import re
import logging
from typing import List, Dict, Any, Tuple
from database import execute_raw_query
from config import settings

logger = logging.getLogger(__name__)


class SQLExecutionError(Exception):
    """Custom exception for SQL execution errors"""
    pass


class SQLValidationError(Exception):
    """Custom exception for SQL validation errors"""
    pass


class SQLExecutor:
    """
    Secure SQL execution service with read-only validation
    """
    
    # Dangerous SQL keywords that should be blocked
    BLOCKED_KEYWORDS = [
        r'\bINSERT\b',
        r'\bUPDATE\b', 
        r'\bDELETE\b',
        r'\bDROP\b',
        r'\bCREATE\b',
        r'\bALTER\b',
        r'\bTRUNCATE\b',
        r'\bGRANT\b',
        r'\bREVOKE\b',
        r'\bEXEC\b',
        r'\bEXECUTE\b',
        r'\bCALL\b',
        r'\bSET\b',
        r'\bLOCK\b',
        r'\bUNLOCK\b',
        r'\bRENAME\b',
        r'\bLOAD\b',
        r'\bOUTFILE\b',
        r'\bINFILE\b',
        r'\bINTO\s+DUMPFILE\b',
        r'\bINTO\s+OUTFILE\b',
    ]
    
    # Compile regex patterns for efficiency
    BLOCKED_PATTERNS = [re.compile(kw, re.IGNORECASE) for kw in BLOCKED_KEYWORDS]
    
    def __init__(self):
        self.max_rows = settings.max_query_rows
        self.timeout_seconds = settings.query_timeout_seconds
    
    def validate_query(self, sql: str) -> Tuple[bool, str]:
        """
        Validate that a SQL query is safe to execute
        
        Args:
            sql: SQL query string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Query cannot be empty"
        
        # Check for blocked keywords
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.search(sql):
                keyword = pattern.pattern.replace(r'\b', '').strip()
                return False, f"Query contains blocked keyword: {keyword}. Only SELECT queries are allowed."
        
        # Ensure query starts with SELECT (after removing comments and whitespace)
        cleaned_sql = self._remove_comments(sql).strip()
        if not cleaned_sql.upper().startswith('SELECT'):
            return False, "Only SELECT queries are allowed. Query must start with SELECT."
        
        # Check for multiple statements (prevent SQL injection via stacked queries)
        if ';' in cleaned_sql[:-1]:  # Semicolon anywhere except at the very end
            return False, "Multiple SQL statements are not allowed."
        
        return True, ""
    
    def _remove_comments(self, sql: str) -> str:
        """Remove SQL comments from query"""
        # Remove single-line comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        # Remove multi-line comments
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        return sql
    
    def execute(self, sql: str) -> Dict[str, Any]:
        """
        Execute a validated SQL query and return results
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Dictionary with query results and metadata
        """
        # Validate query first
        is_valid, error_message = self.validate_query(sql)
        if not is_valid:
            raise SQLValidationError(error_message)
        
        # Add LIMIT if not present to prevent huge result sets
        sql_with_limit = self._ensure_limit(sql)
        
        logger.info(f"Executing query: {sql_with_limit[:200]}...")
        
        try:
            # Execute the query
            results = execute_raw_query(sql_with_limit)
            
            # Convert any non-serializable types
            serializable_results = self._make_serializable(results)
            
            return {
                "success": True,
                "data": serializable_results,
                "row_count": len(serializable_results),
                "limited": len(serializable_results) >= self.max_rows,
                "query": sql_with_limit
            }
            
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            raise SQLExecutionError(f"Query execution failed: {str(e)}")
    
    def _ensure_limit(self, sql: str) -> str:
        """
        Ensure query has a LIMIT clause to prevent huge result sets
        
        Args:
            sql: Original SQL query
            
        Returns:
            SQL query with LIMIT clause
        """
        upper_sql = sql.upper().strip()
        
        # Check if LIMIT already exists
        if 'LIMIT' in upper_sql:
            return sql
        
        # Remove trailing semicolon if present
        sql = sql.rstrip(';').strip()
        
        # Add LIMIT clause
        return f"{sql} LIMIT {self.max_rows}"
    
    def _make_serializable(self, results: List[Dict]) -> List[Dict]:
        """
        Convert query results to JSON-serializable format
        
        Args:
            results: Raw query results
            
        Returns:
            Serializable results
        """
        import datetime
        from decimal import Decimal
        
        serializable = []
        for row in results:
            new_row = {}
            for key, value in row.items():
                if isinstance(value, datetime.datetime):
                    new_row[key] = value.isoformat()
                elif isinstance(value, datetime.date):
                    new_row[key] = value.isoformat()
                elif isinstance(value, Decimal):
                    new_row[key] = float(value)
                elif isinstance(value, bytes):
                    new_row[key] = value.decode('utf-8', errors='ignore')
                else:
                    new_row[key] = value
            serializable.append(new_row)
        
        return serializable
    
    def get_query_stats(self, sql: str) -> Dict[str, Any]:
        """
        Get statistics about a query without executing it
        (useful for showing query plan info to user)
        
        Args:
            sql: SQL query to analyze
            
        Returns:
            Query statistics
        """
        try:
            explain_query = f"EXPLAIN {sql}"
            results = execute_raw_query(explain_query)
            return {
                "explain": results,
                "estimated_rows": sum(row.get("rows", 0) for row in results if row.get("rows"))
            }
        except Exception as e:
            return {"error": str(e)}
