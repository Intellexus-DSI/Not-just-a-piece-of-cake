from typing import Any
import json

class TSVResponseParser:
    """
    A parser for model responses that contain TSV data with optional explanatory text.
    """
    
    def __init__(self, min_columns: int = 2, require_header: bool = True):
        """
        Initialize the parser.
        
        Args:
            min_columns: Minimum number of columns to consider a line as TSV data
            require_header: Whether to require a header row for TSV identification
        """
        self.min_columns = min_columns
        self.require_header = require_header
    
    def parse_response(self, response: str) -> dict[str, Any]:
        """
        Parse a model response containing TSV data.
        
        Args:
            response: The raw response string (can be JSON or plain text)
            
        Returns:
            Dictionary containing parsed data with keys:
            - 'tsv_data': List of rows (each row is a list of columns)
            - 'header': Header row if found
            - 'explanations': Non-TSV text
            - 'raw_tsv': Raw TSV text
        """
        
        # Extract text from JSON if needed
        text = self._extract_text_from_response(response)
        
        # Split into lines
        lines = text.split('\n')
        # Normalize \\t to \t
        lines = [line.replace('\\t', '\t') for line in lines]
        # Identify TSV lines and explanations
        tsv_lines = []
        explanation_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this looks like a TSV line
            if self._is_tsv_line(line):
                tsv_lines.append(line)
            else:
                explanation_lines.append(line)
        
        # Parse TSV data
        parsed_tsv = []
        header = None
        
        for i, line in enumerate(tsv_lines):
            columns = line.split('\t')
            if i == 0 and self.require_header:
                header = columns
            parsed_tsv.append(columns)
        
        # Prepare result
        result = {
            'tsv_data': parsed_tsv,
            'header': header,
            'raw_tsv': '\n'.join(tsv_lines)
        }
        
        result['explanations'] = explanation_lines
        
        return result
    
    def _extract_text_from_response(self, response: str) -> str:
        """Extract text from response, handling JSON format if present."""
        response = response.strip()
        
        # Try to parse as JSON first
        try:
            if response.startswith('{') or response.startswith('"'):
                parsed = json.loads(response)
                if isinstance(parsed, dict) and 'response' in parsed:
                    return parsed['response']
                elif isinstance(parsed, str):
                    return parsed
        except json.JSONDecodeError:
            pass
        
        return response
    
    def _is_tsv_line(self, line: str) -> bool:
        """Determine if a line looks like TSV data."""
        if not line.strip():
            return False
        
        # Count tabs
        tab_count = line.count('\t')
        
        # Must have at least one tab and meet minimum column requirement
        if tab_count < 1 or tab_count + 1 < self.min_columns:
            return False
        
        # Additional heuristics
        columns = line.split('\t')
        
        # Check if it looks like typical TSV patterns
        # - No tabs should result in empty columns (after strip)
        if any(not col.strip() and col != '' for col in columns):
            return False
        
        # If it contains common sentence patterns, it's likely explanation
        sentence_indicators = [
            'it will', 'it\'ll', 'this is', 'the following', 'here is', 'here are',
            'for example', 'note that', 'please', 'you can', 'we can'
        ]
        
        line_lower = line.lower()
        if any(indicator in line_lower for indicator in sentence_indicators):
            # But make exception if it clearly has tabular structure
            if tab_count >= 2:
                return True
            return False
        
        return True
    
    def to_csv(self, parsed_data: dict[str, Any], delimiter: str = ',') -> str:
        """Convert parsed TSV data to CSV format."""
        lines = []
        for row in parsed_data['tsv_data']:
            # Escape fields that contain the delimiter or quotes
            escaped_row = []
            for field in row:
                if delimiter in field or '"' in field or '\n' in field:
                    escaped_field = '"' + field.replace('"', '""') + '"'
                else:
                    escaped_field = field
                escaped_row.append(escaped_field)
            lines.append(delimiter.join(escaped_row))
        
        return '\n'.join(lines)
    
    def to_dataframe(self, parsed_data: dict[str, Any]):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
            data = parsed_data['tsv_data']
            if parsed_data['header'] and len(data) > 1:
                return pd.DataFrame(data[1:], columns=data[0])
            else:
                return pd.DataFrame(data)
        except ImportError:
            raise ImportError("pandas is required for DataFrame conversion")
        
    def get_tags(self, parsed_data: dict[str, Any]) -> list[str]:
        """
        Extract tags from the TSV data.
        
        Args:
            parsed_data: The parsed response data containing 'tsv_data'.
        
        Returns:
            List of unique tags found in the TSV data.
        """
        # Parse the TSV data to df
        df = self.to_dataframe(parsed_data)
        if df.empty:
            return []
        # Assuming the last column contains tags
        tags_column = df.columns[-1] if df.shape[1] > 0 else None
        if tags_column is None:
            return []
        tags = df[tags_column].to_list()
        return tags
    


# Example usage and testing
def test_tsv_parser():
    
    # Also test with the actual content
    actual_content = "It'll be a cold day in hell when that happens.\n\nWord\\tIdiom\nIt\\tO\n'll\\tO\nbe\\tO\na\\tO\ncold\\tO\nday\\tO\nin\\tO\nhell\\tB-IDIOM\nwhen\\tI-IDIOM\nthat\\tI-IDIOM\nhappens\\tI-IDIOM\n.\\tO"
    
    parser = TSVResponseParser()
    
    print("=== Test 1 ===")
    result = parser.parse_response(actual_content)
    print("Explanations:", result.get('explanations', []))
    print("TSV Data (first 5 rows):")
    for i, row in enumerate(result['tsv_data'][:5]):
        print(f"  {row}")
    print(f"Total rows: {len(result['tsv_data'])}")
    
    print("\n=== Test 2: Convert to CSV ===")
    csv_output = parser.to_csv(result)
    print("CSV format (first 200 chars):")
    print(csv_output[:200] + "..." if len(csv_output) > 200 else csv_output)
    
    print("\n=== Raw TSV ===")
    print(result['raw_tsv'][:200] + "..." if len(result['raw_tsv']) > 200 else result['raw_tsv'])

    print("\n=== Test 3: Convert to DataFrame ===")
    df = parser.to_dataframe(result)
    print("DataFrame (first 5 rows):")
    print(df.head())

    print("\n=== Test 4: Extract Tags ===")
    tags = parser.get_tags(result)
    print("Extracted Tags:", tags)

if __name__ == "__main__":
    test_tsv_parser()