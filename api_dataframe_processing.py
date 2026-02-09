import pandas as pd
import requests
import json
from typing import List, Dict, Any
import time
from tqdm import tqdm  # For progress bar
import logging
from datetime import datetime
# ============================================================================
# SETUP LOGGING (Optional but recommended)
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# METHOD 1: SIMPLE API CALL & FLATTEN JSON (Basic Approach)
# ============================================================================

def api_call_simple(df: pd.DataFrame, api_endpoint: str, key_column: str) -> pd.DataFrame:
    """
    Simple row-by-row API calls and flatten JSON response.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with data to send to API
    api_endpoint : str
        API endpoint URL (e.g., 'https://api.example.com/data')
    key_column : str
        Column name to use as API parameter
    
    Returns:
    --------
    pd.DataFrame
        New dataframe with original data + flattened API responses
    """
    
    results = []
    
    # Iterate through each row
    for idx, row in df.iterrows():
        try:
            # Extract value from the specified column
            query_param = row[key_column]
            
            # Make API call
            response = requests.get(
                api_endpoint,
                params={'query': query_param},
                timeout=10
            )
            
            # Check if request was successful
            if response.status_code == 200:
                api_data = response.json()
                
                # Flatten the JSON response
                flattened = flatten_json(api_data)
                
                # Combine original row data + flattened API response
                result_row = {**row.to_dict(), **flattened}
                results.append(result_row)
                
                logger.info(f"Row {idx}: Successfully processed")
            else:
                logger.error(f"Row {idx}: API returned status {response.status_code}")
                # Optionally: add row with error info
                result_row = {**row.to_dict(), 'api_error': f"Status {response.status_code}"}
                results.append(result_row)
        
        except Exception as e:
            logger.error(f"Row {idx}: Error - {str(e)}")
            result_row = {**row.to_dict(), 'api_error': str(e)}
            results.append(result_row)
    
    # Convert list of dictionaries to DataFrame
    result_df = pd.DataFrame(results)
    return result_df

# ============================================================================
# METHOD 2: WITH PROGRESS BAR & RATE LIMITING (Production Ready)
# ============================================================================

def api_call_with_progress(
    df: pd.DataFrame,
    api_endpoint: str,
    key_column: str,
    rate_limit_delay: float = 1.0,
    timeout: int = 10
) -> pd.DataFrame:
    """
    API calls with progress bar and rate limiting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    api_endpoint : str
        API endpoint URL
    key_column : str
        Column name to use as API parameter
    rate_limit_delay : float
        Delay between API calls in seconds (respects API rate limits)
    timeout : int
        Request timeout in seconds
    
    Returns:
    --------
    pd.DataFrame
        Result dataframe with API responses
    """
    
    results = []
    
    # Use tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        try:
            query_param = row[key_column]
            
            # Make API call
            response = requests.get(
                api_endpoint,
                params={'query': query_param},
                timeout=timeout,
                headers={'User-Agent': 'pandas-api-client/1.0'}
            )
            
            if response.status_code == 200:
                api_data = response.json()
                flattened = flatten_json(api_data)
                result_row = {**row.to_dict(), **flattened}
            else:
                result_row = {**row.to_dict(), 'api_error': f"Status {response.status_code}"}
            
            results.append(result_row)
            
            # Rate limiting - wait between requests
            time.sleep(rate_limit_delay)
        
        except requests.exceptions.Timeout:
            logger.warning(f"Row {idx}: Request timeout")
            result_row = {**row.to_dict(), 'api_error': 'Timeout'}
            results.append(result_row)
        
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Row {idx}: Connection error - {str(e)}")
            result_row = {**row.to_dict(), 'api_error': 'Connection error'}
            results.append(result_row)
        
        except Exception as e:
            logger.error(f"Row {idx}: {str(e)}")
            result_row = {**row.to_dict(), 'api_error': str(e)}
            results.append(result_row)
    
    result_df = pd.DataFrame(results)
    return result_df

# ============================================================================
# METHOD 3: WITH CUSTOM REQUEST BUILDER (Most Flexible)
# ============================================================================

def api_call_custom_requests(
    df: pd.DataFrame,
    api_endpoint: str,
    request_builder_func,
    rate_limit_delay: float = 1.0
) -> pd.DataFrame:
    """
    API calls with custom request builder function.
    Allows complex request construction per row.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    api_endpoint : str
        API endpoint URL
    request_builder_func : callable
        Function that takes a row and returns request parameters dict
        Example: lambda row: {'params': {'id': row['id'], 'type': row['type']}}
    rate_limit_delay : float
        Delay between API calls
    
    Returns:
    --------
    pd.DataFrame
        Result dataframe with API responses
    """
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        try:
            # Use custom function to build request parameters
            request_config = request_builder_func(row)
            
            # Make API call with custom parameters
            response = requests.get(api_endpoint, **request_config, timeout=10)
            
            if response.status_code == 200:
                api_data = response.json()
                flattened = flatten_json(api_data)
                result_row = {**row.to_dict(), **flattened}
            else:
                result_row = {**row.to_dict(), 'api_error': f"Status {response.status_code}"}
            
            results.append(result_row)
            time.sleep(rate_limit_delay)
        
        except Exception as e:
            logger.error(f"Row {idx}: {str(e)}")
            result_row = {**row.to_dict(), 'api_error': str(e)}
            results.append(result_row)
    
    return pd.DataFrame(results)

# ============================================================================
# METHOD 4: BATCH PROCESSING (For Large Datasets - Most Efficient)
# ============================================================================

def api_call_batch(
    df: pd.DataFrame,
    api_endpoint: str,
    batch_size: int = 10,
    rate_limit_delay: float = 1.0
) -> pd.DataFrame:
    """
    Process API calls in batches for efficiency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    api_endpoint : str
        API endpoint that accepts batch requests
    batch_size : int
        Number of rows per batch
    rate_limit_delay : float
        Delay between batch requests
    
    Returns:
    --------
    pd.DataFrame
        Result dataframe
    """
    
    results = []
    
    # Split dataframe into batches
    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]
    
    for batch_idx, batch_df in tqdm(enumerate(batches), total=len(batches), desc="Processing batches"):
        try:
            # Prepare batch data (convert batch to list of dicts)
            batch_data = batch_df.to_dict('records')
            
            # Send entire batch to API
            response = requests.post(
                api_endpoint,
                json={'items': batch_data},  # Send as JSON
                timeout=30
            )
            
            if response.status_code == 200:
                batch_results = response.json()  # Assuming API returns list
                
                # Process each result in the batch
                for original_row, api_result in zip(batch_df.to_dict('records'), batch_results):
                    flattened = flatten_json(api_result)
                    result_row = {**original_row, **flattened}
                    results.append(result_row)
            else:
                logger.error(f"Batch {batch_idx}: API returned {response.status_code}")
                for original_row in batch_df.to_dict('records'):
                    result_row = {**original_row, 'api_error': f"Status {response.status_code}"}
                    results.append(result_row)
            
            time.sleep(rate_limit_delay)
        
        except Exception as e:
            logger.error(f"Batch {batch_idx}: {str(e)}")
            for original_row in batch_df.to_dict('records'):
                result_row = {**original_row, 'api_error': str(e)}
                results.append(result_row)
    
    return pd.DataFrame(results)

# ============================================================================
# METHOD 5: PARALLEL PROCESSING (For Maximum Speed - Advanced)
# ============================================================================

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def api_call_parallel(
    df: pd.DataFrame,
    api_endpoint: str,
    key_column: str,
    max_workers: int = 5
) -> pd.DataFrame:
    """
    Parallel API calls using ThreadPoolExecutor.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    api_endpoint : str
        API endpoint
    key_column : str
        Column to use as parameter
    max_workers : int
        Number of parallel threads
    
    Returns:
    --------
    pd.DataFrame
        Result dataframe
    """
    
    results = {}
    results_lock = Lock()
    
    def process_row(idx, row):
        """Process single row with API call"""
        try:
            query_param = row[key_column]
            response = requests.get(
                api_endpoint,
                params={'query': query_param},
                timeout=10
            )
            
            if response.status_code == 200:
                api_data = response.json()
                flattened = flatten_json(api_data)
                result_row = {**row.to_dict(), **flattened}
            else:
                result_row = {**row.to_dict(), 'api_error': f"Status {response.status_code}"}
            
            return idx, result_row
        
        except Exception as e:
            logger.error(f"Row {idx}: {str(e)}")
            return idx, {**row.to_dict(), 'api_error': str(e)}
    
    # Execute with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_row, idx, row): idx 
            for idx, row in df.iterrows()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
            idx, result_row = future.result()
            with results_lock:
                results[idx] = result_row
    
    # Reconstruct dataframe in original order
    result_list = [results[i] for i in sorted(results.keys())]
    return pd.DataFrame(result_list)

# ============================================================================
# UTILITY: FLATTEN JSON FUNCTION
# ============================================================================

def flatten_json(nested_json: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten nested JSON structure.
    
    Parameters:
    -----------
    nested_json : dict
        Nested JSON/dictionary
    parent_key : str
        Parent key prefix (used recursively)
    sep : str
        Separator for nested keys
    
    Returns:
    --------
    dict
        Flattened dictionary
    
    Example:
    --------
    >>> data = {'user': {'name': 'John', 'age': 30}, 'email': 'john@example.com'}
    >>> flatten_json(data)
    {'user_name': 'John', 'user_age': 30, 'email': 'john@example.com'}
    """
    items = []
    
    for k, v in nested_json.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Handle lists - can either flatten or convert to string
            items.append((new_key, json.dumps(v)))  # Convert list to JSON string
        else:
            items.append((new_key, v))
    
    return dict(items)

# ============================================================================
# PRACTICAL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: Create sample input dataframe
    print("="*80)
    print("EXAMPLE 1: Basic API Calls Row-by-Row")
    print("="*80)
    
    sample_df = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'username': ['alice', 'bob', 'charlie', 'david', 'eve'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'david@example.com', 'eve@example.com']
    })
    
    print("\n📥 Input DataFrame:")
    print(sample_df)
    
    # Example 2: Using mock API (for demonstration)
    print("\n" + "="*80)
    print("EXAMPLE 2: Mock API Call (Demonstration)")
    print("="*80)
    
    def mock_api_call(df: pd.DataFrame, key_column: str) -> pd.DataFrame:
        """Mock API that simulates real API responses"""
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            # Simulate API response
            api_response = {
                'user_id': row['user_id'],
                'profile': {
                    'bio': f"Bio for {row['username']}",
                    'followers': 100 + (idx * 10),
                    'posts': {
                        'count': 5 + idx,
                        'recent': ['post1', 'post2', 'post3']
                    }
                },
                'metadata': {
                    'last_updated': '2025-02-09',
                    'verified': True
                }
            }
            
            flattened = flatten_json(api_response)
            result_row = {**row.to_dict(), **flattened}
            results.append(result_row)
            time.sleep(0.5)  # Simulate API delay
        
        return pd.DataFrame(results)
    
    result_df = mock_api_call(sample_df, 'username')
    
    print("\n📤 Output DataFrame with API results:")
    print(result_df)
    print(f"\nShape: {result_df.shape}")
    print(f"\nColumns: {list(result_df.columns)}")
    
    result_df['last_updated'] = datetime.now().strftime('%Y-%m-%d')
    # Example 3: Custom request builder
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Request Builder")
    print("="*80)
    
    def custom_request_builder(row):
        """Build custom request parameters per row"""
        return {
            'params': {
                'id': row['user_id'],
                'username': row['username'],
                'include_details': True
            }
        }
    
    # This would be used with: api_call_custom_requests(sample_df, endpoint, custom_request_builder)
    print("\n✓ Custom request builder defined")
    print("  Usage: api_call_custom_requests(df, endpoint, custom_request_builder)")
    
    # Example 4: Error handling
    print("\n" + "="*80)
    print("EXAMPLE 4: Error Handling & Data Quality")
    print("="*80)
    
    result_df_with_errors = result_df.copy()
    
    # Check for errors
    if 'api_error' in result_df_with_errors.columns:
        errors = result_df_with_errors[result_df_with_errors['api_error'].notna()]
        print(f"\n⚠️  Rows with errors: {len(errors)}")
        if len(errors) > 0:
            print(errors[['user_id', 'api_error']])
    else:
        print("\n✓ No API errors encountered")
    
    # Example 5: Save results
    print("\n" + "="*80)
    print("EXAMPLE 5: Save Results")
    print("="*80)
    
    result_df.to_csv('api_results.csv', index=False)
    print("✓ Results saved to 'api_results.csv'")
    
    result_df.to_json('api_results.json', orient='records')
    print("✓ Results saved to 'api_results.json'")
    
    # Example 6: Summary statistics
    print("\n" + "="*80)
    print("EXAMPLE 6: Summary Statistics")
    print("="*80)
    
    print(f"Total rows processed: {len(result_df)}")
    print(f"Original columns: {len(sample_df.columns)}")
    print(f"Final columns: {len(result_df.columns)}")
    print(f"New columns added: {len(result_df.columns) - len(sample_df.columns)}")