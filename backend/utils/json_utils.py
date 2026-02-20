import json
import re
from typing import Any, Dict
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def safe_json_parse(text: str) -> Dict[str, Any]:
    """
    Robustly parse JSON from LLM output, handling markdown blocks and illegal control characters.
    """
    if not text:
        return {}
    
    # 1. Strip whitespace
    text = text.strip()
    
    # 2. Extract content from markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    # 3. Remove illegal control characters (ASCII 0-31) that are not escaped.
    # JSON strings cannot contain raw control characters. 
    # We replace them with spaces or remove them.
    # Note: \n, \r, \t are often the culprits if they are raw and not escaped.
    
    # This regex finds raw control characters except for those that might be escaped (though json.loads handles escapes)
    # Most common issue is raw newlines in a string.
    
    # First, let's try a direct parse with strict=False.
    # strict=False allows literal control characters (like newlines) inside strings.
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        logger.debug(f"Initial JSON parse failed: {e}. Attempting cleanup.")
        
        # Cleanup: Remove markdown code blocks again just in case
        text = re.sub(r'```json\s*|\s*```', '', text).strip()
        
        # Extract everything between the first '{' and the last '}'
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                candidate = text[start:end+1]
                return json.loads(candidate, strict=False)
        except Exception as inner_e:
            logger.debug(f"Cleanup extraction failed: {inner_e}")
        
        # If it still fails, try to replace common issues
        try:
            logger.error(f"JSON parsing failed after cleanup. Problematic text:\n{text}")
            # Replace unescaped backslashes (that are not part of a valid escape)
            raise e
        except:
            raise e
