from openai import OpenAI, AsyncOpenAI
from typing import Optional, Dict, List
import json
import os
import argparse
from datasets import load_from_disk
from tqdm import tqdm
from datetime import datetime
import asyncio
import time
import re

# Kimi API base URL
KIMI_BASE_URL = "https://api.moonshot.ai/v1"

# GLM API base URL (via Together.ai)
GLM_BASE_URL = "https://api.together.xyz/v1"

# Try to import the new Batch API SDK (optional)
try:
    from google import genai as genai_batch
    BATCH_API_AVAILABLE = True
except ImportError:
    BATCH_API_AVAILABLE = False

# OpenAI Batch API is available via the standard openai package
OPENAI_BATCH_API_AVAILABLE = True

def extract_answer(text: str) -> Optional[bool]:
    """Extract yes/no answer from generated text.

    Handles various formats including markdown formatting like **Yes** or *No*.
    """
    text_lower = text.lower()

    # Remove markdown bold/italic markers for matching
    # This handles **yes**, *yes*, __yes__, _yes_, etc.
    text_cleaned = re.sub(r'[*_]+', '', text_lower)

    # Check for "therefore: yes/no" or "answer: yes/no" patterns
    if "therefore: yes" in text_cleaned or "answer: yes" in text_cleaned:
        return True
    elif "therefore: no" in text_cleaned or "answer: no" in text_cleaned:
        return False

    # Also check for patterns with whitespace variations (e.g., "therefore:  yes")
    if re.search(r'therefore:\s*yes', text_cleaned):
        return True
    elif re.search(r'therefore:\s*no', text_cleaned):
        return False

    if re.search(r'answer:\s*yes', text_cleaned):
        return True
    elif re.search(r'answer:\s*no', text_cleaned):
        return False

    return None


def build_prediction_dict(sample: Dict, index: int, prompt: str, ground_truth: bool,
                          predicted: bool, generated_text: str,
                          finish_reason: str = None, token_usage: Dict = None,
                          error: str = None) -> Dict:
    """Build a prediction dictionary for binary label classification."""

    is_correct = (predicted == ground_truth) if predicted is not None else False

    pred_dict = {
            'index': index,
            'input': prompt,
            'ground_truth_label': ground_truth,
            'predicted_answer': predicted,
            'is_correct': is_correct,
            'generated_text': generated_text,
    }

    if error:
        pred_dict['error'] = error

    if finish_reason:
        pred_dict['finish_reason'] = finish_reason

    if token_usage:
        pred_dict['token_usage'] = token_usage

    # Add any available metadata from the sample
    metadata_keys = ['num_variables', 'template', 'rung', 'query_type', 'graph_id', 'story_id', 'dataset', 'num_edges']
    for key in metadata_keys:
        if key in sample and sample.get(key) is not None:
            pred_dict[key] = sample[key]

    return pred_dict

def is_reasoning_model(model_name: str) -> bool:
    """Check if the model is an OpenAI model that supports reasoning_effort.

    Models that support reasoning_effort:
    - o1, o3 series (reasoning models)
    - GPT-5, GPT-5.1, GPT-5.2 (support none/low/medium/high/xhigh)
    - gpt-oss-120b (OpenAI's open-weight reasoning model)

    Note: GPT-5 defaults to 'medium', GPT-5.1/5.2 default to 'none'.
    """
    reasoning_prefixes = ('o1', 'o3', 'gpt-5', 'gpt-oss')
    return any(model_name.startswith(prefix) or f'/{prefix}' in model_name for prefix in reasoning_prefixes)

def uses_max_completion_tokens(model_name: str) -> bool:
    """Check if the model uses max_completion_tokens instead of max_tokens.

    Newer OpenAI models (GPT-5, o1, o3, gpt-oss series) require max_completion_tokens.
    Older models (GPT-4, GPT-4o, etc.) use max_tokens.
    """
    # Models that use max_completion_tokens
    new_style_prefixes = ('o1', 'o3', 'gpt-5', 'gpt-oss')
    return any(model_name.startswith(prefix) or f'/{prefix}' in model_name for prefix in new_style_prefixes)

def batch_call_openai_official(model_name: str, prompts: List[str], api_key: str,
                                reasoning_effort: str = "medium", poll_interval: int = 60):
    """
    Use OpenAI's official Batch API for cost-effective batch processing (50% cost reduction).

    This function creates a batch job by uploading a JSONL file and polls until completion.
    Results are downloaded and parsed, preserving order via custom_id.

    Args:
        model_name: Name of the OpenAI model (e.g., 'gpt-4o', 'o1-mini')
        prompts: List of prompts to process
        api_key: OpenAI API key
        reasoning_effort: Reasoning effort level for reasoning models (default: "medium")
        poll_interval: How often to check job status in seconds (default: 60)

    Returns:
        List of (generated_text, finish_reason, token_usage) tuples in the same order as prompts
    """
    client = OpenAI(api_key=api_key)

    # Create temporary directory for batch files
    batch_dir = "./evaluation_results/batch_raw_results"
    os.makedirs(batch_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file_path = os.path.join(batch_dir, f"openai_batch_input_{timestamp}.jsonl")

    # Prepare JSONL input file
    print(f"\nPreparing batch request with {len(prompts)} prompts...")

    with open(input_file_path, 'w') as f:
        for idx, prompt in enumerate(prompts):
            # Build body parameters based on model type
            body = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Only set temperature for non-reasoning models (reasoning models only support default temperature=1)
            if not is_reasoning_model(model_name):
                body["temperature"] = 0  # Greedy decoding for reproducibility

            # Set appropriate token limits based on model
            if uses_max_completion_tokens(model_name):
                body["max_completion_tokens"] = 60000
            elif 'gpt-4o' in model_name or 'gpt-4-turbo' in model_name:
                body["max_tokens"] = 16384  # GPT-4o max output is 16K
            elif model_name == 'gpt-4' or model_name.startswith('gpt-4-0'):
                body["max_tokens"] = 4096  # Base GPT-4 has only 8K context total
            else:
                body["max_tokens"] = 8192  # Default

            # Only add reasoning_effort for reasoning models (o1, o3 series)
            if is_reasoning_model(model_name):
                body["reasoning_effort"] = reasoning_effort

            request = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }
            f.write(json.dumps(request) + "\n")

    # Upload the input file
    print(f"Uploading batch input file...")
    with open(input_file_path, 'rb') as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )

    print(f"File uploaded: {batch_input_file.id}")

    # Create the batch job
    print(f"Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"evaluation-{timestamp}"
        }
    )

    print(f"Batch job created: {batch_job.id}")
    print(f"Initial status: {batch_job.status}")
    print("Waiting for results...")

    # Monitor job status
    completed_statuses = {'completed', 'failed', 'expired', 'cancelled'}
    start_time = time.time()

    while batch_job.status not in completed_statuses:
        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        # Show progress info
        progress_info = ""
        if batch_job.request_counts:
            counts = batch_job.request_counts
            progress_info = f" (completed: {counts.completed}/{counts.total}, failed: {counts.failed})"

        print(f"[{elapsed_str}] Status: {batch_job.status}{progress_info}")
        time.sleep(poll_interval)
        batch_job = client.batches.retrieve(batch_job.id)

    total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"\nBatch job completed in {total_time}")
    print(f"Final status: {batch_job.status}")

    if batch_job.status == 'completed':
        # Check if there's an error file (indicates failures)
        if batch_job.error_file_id:
            print(f"Downloading error file to diagnose failures...")
            error_content = client.files.content(batch_job.error_file_id)
            error_file_path = os.path.join(batch_dir, f"openai_batch_errors_{timestamp}.jsonl")
            with open(error_file_path, 'wb') as f:
                f.write(error_content.content)
            print(f"Error file saved to: {error_file_path}")

            # Print first few errors to help diagnose
            error_lines = error_content.content.decode('utf-8').strip().split('\n')[:5]
            print(f"\nFirst {len(error_lines)} errors:")
            for line in error_lines:
                if line:
                    try:
                        err_data = json.loads(line)
                        print(f"  - {err_data.get('error', err_data)}")
                    except:
                        print(f"  - {line[:200]}")

        # Download the output file
        if not batch_job.output_file_id:
            # If no output file but we have error file, all requests failed
            if batch_job.error_file_id:
                raise RuntimeError(f"All requests failed. Check error file: {error_file_path}")
            raise RuntimeError(f"No output file found for completed batch job: {batch_job.id}")

        print(f"Downloading results...")
        output_content = client.files.content(batch_job.output_file_id)

        # Save raw output for recovery
        output_file_path = os.path.join(batch_dir, f"openai_batch_output_{timestamp}.jsonl")
        with open(output_file_path, 'wb') as f:
            f.write(output_content.content)

        print(f"Raw results saved to: {output_file_path}")

        # Parse results and restore order
        results_dict = {}
        for line in output_content.content.decode('utf-8').strip().split('\n'):
            if not line:
                continue
            response_data = json.loads(line)
            custom_id = response_data.get('custom_id', '')

            # Extract index from custom_id (format: "request-{idx}")
            try:
                idx = int(custom_id.split('-')[1])
            except (IndexError, ValueError):
                print(f"Warning: Could not parse custom_id: {custom_id}")
                continue

            # Check for errors
            if response_data.get('error'):
                error_msg = response_data['error'].get('message', 'Unknown error')
                results_dict[idx] = (None, None, {}, error_msg)
                continue

            # Extract response data
            response_body = response_data.get('response', {}).get('body', {})

            generated_text = None
            finish_reason = None
            token_usage = {}

            if 'choices' in response_body and response_body['choices']:
                choice = response_body['choices'][0]
                generated_text = choice.get('message', {}).get('content')
                finish_reason = choice.get('finish_reason')

            if 'usage' in response_body:
                usage = response_body['usage']
                token_usage = {
                    'prompt_tokens': usage.get('prompt_tokens'),
                    'completion_tokens': usage.get('completion_tokens'),
                    'total_tokens': usage.get('total_tokens')
                }

            results_dict[idx] = (generated_text, finish_reason, token_usage)

        # Restore order based on original prompt indices
        results = []
        for idx in range(len(prompts)):
            if idx in results_dict:
                results.append(results_dict[idx])
            else:
                print(f"Warning: Missing result for request {idx}")
                results.append((None, None, {}, f"Missing result for request {idx}"))

        print(f"Successfully processed {len(results)} results")

        # Check for error file
        if batch_job.error_file_id:
            print(f"Note: Some requests had errors. Error file: {batch_job.error_file_id}")

        return results
    else:
        error_msg = f"Batch job failed with status: {batch_job.status}"
        if batch_job.errors:
            error_msg += f"\nErrors: {batch_job.errors}"
        raise RuntimeError(error_msg)


def call_kimi_model(client: OpenAI, model_name: str, prompt: str):
    """Call Kimi API and return standardized response.

    Kimi K2 Thinking is a reasoning model that uses chain-of-thought internally.
    The API is OpenAI-compatible.
    """
    # Build request parameters for Kimi
    params = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60000,  # Kimi supports long context (up to 128K)
        "temperature": 0,  # Greedy decoding for reproducibility
    }

    response = client.chat.completions.create(**params)

    finish_reason = response.choices[0].finish_reason
    generated_text = response.choices[0].message.content

    token_usage = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }

    return generated_text, finish_reason, token_usage

async def call_kimi_model_async(client: AsyncOpenAI, model_name: str, prompt: str):
    """Call Kimi API asynchronously and return standardized response."""
    # Build request parameters for Kimi
    params = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60000,  # Kimi supports long context (up to 128K)
        "temperature": 0,  # Greedy decoding for reproducibility
    }

    response = await client.chat.completions.create(**params)

    finish_reason = response.choices[0].finish_reason
    generated_text = response.choices[0].message.content

    token_usage = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }

    return generated_text, finish_reason, token_usage

async def batch_call_kimi(client: AsyncOpenAI, model_name: str, prompts: List[str], max_concurrent: int = 10):
    """Call Kimi API for multiple prompts concurrently, preserving order."""
    semaphore = asyncio.Semaphore(max_concurrent)

    completed_count = [0]
    pbar = tqdm(total=len(prompts), desc="Processing batch")

    async def call_with_semaphore(prompt):
        async with semaphore:
            try:
                result = await call_kimi_model_async(client, model_name, prompt)
                completed_count[0] += 1
                pbar.update(1)
                return result
            except Exception as e:
                completed_count[0] += 1
                pbar.update(1)
                return None, None, {}, str(e)

    tasks = [call_with_semaphore(prompt) for prompt in prompts]

    # Use gather to preserve order
    results = await asyncio.gather(*tasks)
    pbar.close()

    return results


def call_glm_model(client: OpenAI, model_name: str, prompt: str):
    """Call GLM API via Together.ai and return standardized response.

    GLM models are accessed through Together.ai's OpenAI-compatible API.
    Supported models: zai-org/GLM-4.7, zai-org/GLM-4.7-Flash, etc.
    """
    # Build request parameters for GLM
    params = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60000,  # GLM-4.7 supports up to 128K output
        "temperature": 0,  # Greedy decoding for reproducibility
    }

    response = client.chat.completions.create(**params)

    finish_reason = response.choices[0].finish_reason
    generated_text = response.choices[0].message.content

    token_usage = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }

    return generated_text, finish_reason, token_usage


async def call_glm_model_async(client: AsyncOpenAI, model_name: str, prompt: str, max_retries: int = 5):
    """Call GLM API asynchronously and return standardized response.

    Includes retry logic with exponential backoff for transient errors (503, 429, etc.)
    """
    # Build request parameters for GLM/GPT-OSS via Together.ai
    params = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60000,  # Together.ai models support large outputs
        "temperature": 0,  # Greedy decoding for reproducibility
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(**params)

            finish_reason = response.choices[0].finish_reason
            generated_text = response.choices[0].message.content

            token_usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            return generated_text, finish_reason, token_usage
        except Exception as e:
            last_error = e
            error_str = str(e)
            # Retry on transient errors (503 service unavailable, 429 rate limit, 500 server error)
            if '503' in error_str or '429' in error_str or '500' in error_str or 'service_unavailable' in error_str.lower():
                wait_time = (2 ** attempt) + (asyncio.get_event_loop().time() % 1)  # Exponential backoff with jitter
                await asyncio.sleep(wait_time)
                continue
            else:
                raise  # Non-retryable error, raise immediately

    # All retries exhausted
    raise last_error


async def batch_call_glm(client: AsyncOpenAI, model_name: str, prompts: List[str], max_concurrent: int = 10):
    """Call GLM API for multiple prompts concurrently, preserving order."""
    semaphore = asyncio.Semaphore(max_concurrent)

    completed_count = [0]
    pbar = tqdm(total=len(prompts), desc="Processing batch")

    async def call_with_semaphore(prompt):
        async with semaphore:
            try:
                result = await call_glm_model_async(client, model_name, prompt)
                completed_count[0] += 1
                pbar.update(1)
                return result
            except Exception as e:
                completed_count[0] += 1
                pbar.update(1)
                return None, None, {}, str(e)

    tasks = [call_with_semaphore(prompt) for prompt in prompts]

    # Use gather to preserve order
    results = await asyncio.gather(*tasks)
    pbar.close()

    return results


def batch_call_gemini_official(model_name: str, prompts: List[str], api_key: str,
                                poll_interval: int = 60, thinking_level: str = None):
    """
    Use Gemini's official Batch API for cost-effective batch processing (50% cost reduction).

    This function creates a batch job using file-based requests with unique keys to ensure
    proper correlation between requests and responses (inline requests do not preserve order).

    Args:
        model_name: Name of the Gemini model (e.g., 'gemini-3-pro-preview')
        prompts: List of prompts to process
        api_key: Google API key
        poll_interval: How often to check job status in seconds (default: 60)
        thinking_level: NOTE: Currently not supported in Batch API. Gemini 3 models
                       default to "high" thinking which cannot be changed in batch mode.

    Returns:
        List of (generated_text, finish_reason, token_usage) tuples in the same order as prompts
    """
    if not BATCH_API_AVAILABLE:
        raise ImportError("google-genai package is required for Batch API. Install with: pip install google-genai")

    client = genai_batch.Client(api_key=api_key)

    if thinking_level and thinking_level.lower() != "high":
        print(f"WARNING: thinking_level='{thinking_level}' requested, but Batch API does not support")
        print(f"         custom thinking levels. Gemini 3 will use default 'high' thinking.")

    # Create temporary directory for batch files
    batch_dir = "./evaluation_results/batch_raw_results"
    os.makedirs(batch_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file_path = os.path.join(batch_dir, f"gemini_batch_input_{timestamp}.jsonl")

    # Prepare JSONL input file with unique keys for each request
    print(f"\nPreparing batch request with {len(prompts)} prompts...")
    print(f"Note: Gemini 3 Batch API uses default 'high' thinking (not configurable)")

    with open(input_file_path, 'w') as f:
        for idx, prompt in enumerate(prompts):
            request = {
                "key": f"request-{idx}",
                "request": {
                    "contents": [{
                        "parts": [{"text": prompt}],
                        "role": "user"
                    }]
                }
            }
            f.write(json.dumps(request) + "\n")

    # Upload the input file to Gemini Files API
    print(f"Uploading batch input file...")
    with open(input_file_path, 'rb') as f:
        uploaded_file = client.files.upload(
            file=f,
            config={
                'display_name': f"batch_input_{timestamp}.jsonl",
                'mime_type': 'application/jsonl'
            }
        )

    print(f"File uploaded: {uploaded_file.name}")

    # Create batch job with file-based input
    batch_job = client.batches.create(
        model=f"models/{model_name}",
        src=uploaded_file.name,
        config={'display_name': f"evaluation-{timestamp}"}
    )

    print(f"Batch job created: {batch_job.name}")
    print(f"Initial state: {batch_job.state}")
    print("Waiting for results...")

    # Monitor job status
    completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'}
    start_time = time.time()

    while batch_job.state.name not in completed_states:
        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(f"[{elapsed_str}] Current state: {batch_job.state.name}")
        time.sleep(poll_interval)

        # Retry logic for transient network errors
        max_retries = 15
        retry_delay = 10
        for attempt in range(max_retries):
            try:
                batch_job = client.batches.get(name=batch_job.name)
                break
            except Exception as e:
                error_msg = str(e).lower()
                if 'getaddrinfo' in error_msg or 'connect' in error_msg or 'network' in error_msg:
                    if attempt < max_retries - 1:
                        print(f"  Network error (attempt {attempt + 1}/{max_retries}): {e}")
                        print(f"  Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print(f"  Network error persisted after {max_retries} attempts")
                        raise
                else:
                    raise

    total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"\nBatch job completed in {total_time}")
    print(f"Final state: {batch_job.state.name}")

    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
        # Get the output file from the batch job
        if not (hasattr(batch_job, 'dest') and batch_job.dest and
                hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name):
            raise RuntimeError(f"No output file found in batch job. Job name: {batch_job.name}")

        output_file_name = batch_job.dest.file_name
        print(f"Downloading results from: {output_file_name}")

        # Strip 'files/' prefix if present - the API expects just the file ID
        file_id = output_file_name.replace('files/', '') if output_file_name.startswith('files/') else output_file_name

        # Download file content
        output_file_path = os.path.join(batch_dir, f"gemini_batch_output_{timestamp}.jsonl")

        # Download using client.files.download with file= parameter
        try:
            file_content = client.files.download(file=file_id)
            with open(output_file_path, 'wb') as f:
                f.write(file_content)
        except Exception as e:
            raise RuntimeError(f"Failed to download output file: {e}")

        print(f"Raw results saved to: {output_file_path}")

        # Parse results and restore order using keys
        results_dict = {}
        with open(output_file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                response_data = json.loads(line)
                key = response_data.get('key', '')

                # Extract index from key (format: "request-{idx}")
                try:
                    idx = int(key.split('-')[1])
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse key: {key}")
                    continue

                # Check for errors
                if response_data.get('error'):
                    error_msg = response_data['error'].get('message', str(response_data['error']))
                    results_dict[idx] = (None, None, {}, error_msg)
                    continue

                # Extract response data
                response = response_data.get('response', {})
                generated_text = None
                finish_reason = None
                token_usage = {}

                if 'candidates' in response and response['candidates']:
                    candidate = response['candidates'][0]
                    finish_reason = candidate.get('finishReason', candidate.get('finish_reason'))

                    content = candidate.get('content', {})
                    parts = content.get('parts', [])
                    if parts and 'text' in parts[0]:
                        generated_text = parts[0]['text']

                if 'usageMetadata' in response:
                    usage = response['usageMetadata']
                    token_usage = {
                        'prompt_tokens': usage.get('promptTokenCount'),
                        'completion_tokens': usage.get('candidatesTokenCount'),
                        'total_tokens': usage.get('totalTokenCount')
                    }

                results_dict[idx] = (generated_text, finish_reason, token_usage)

        # Restore order based on original prompt indices
        results = []
        for idx in range(len(prompts)):
            if idx in results_dict:
                results.append(results_dict[idx])
            else:
                print(f"Warning: Missing result for request {idx}")
                results.append((None, None, {}, f"Missing result for request {idx}"))

        print(f"Successfully processed {len(results)} results")

        return results
    else:
        raise RuntimeError(f"Batch job failed with state: {batch_job.state.name}")

def evaluate_model_on_dataset(
    model_name: str,
    dataset,
    max_samples,
    output_file: str,
    api_key: str,
    model_type: str = "gemini",  # "openai", "gemini", "kimi", "glm", or "gpt-oss"
    reasoning_effort: str = "medium",  # OpenAI: "low", "medium", "high"
    thinking_level: str = None,  # Gemini 3: "low", "high"; Gemini 3 Flash: "minimal", "low", "medium", "high"
    beginning: int = 0,
    max_concurrent: int = 10,  # Max concurrent requests for Kimi/GLM/GPT-OSS
    batch_poll_interval: int = 60,  # Polling interval for official batch API (seconds)
) -> Dict:
    
    dataset = dataset.select(range(beginning, min(beginning + max_samples, len(dataset))))

    total = len(dataset)
    correct = 0
    predictions = []
    no_answer_count = 0
    api_errors = 0
    empty_content_count = 0
    length_cutoff_count = 0

    print(f"Evaluating {model_name} on {total} samples")

    def get_ground_truth(sample):
        """Get ground truth label."""
        return bool(sample['label'])

    def get_prediction(generated_text):
        """Extract prediction label."""
        return extract_answer(generated_text)

    def check_correctness(predicted, ground_truth):
        """Check if prediction is correct."""
        return predicted == ground_truth if predicted is not None else False

    # Use official Batch API for Gemini
    if model_type == "gemini":
        if not api_key:
            raise ValueError("API key required for official Batch API")

        print(f"Using official Gemini Batch API (50% cost reduction, may take up to 24 hours)...")
        prompts = [sample["query"] for sample in dataset]

        # Run official batch processing
        batch_results = batch_call_gemini_official(
            model_name=model_name,
            prompts=prompts,
            api_key=api_key,
            poll_interval=batch_poll_interval,
            thinking_level=thinking_level
        )

        # Process batch results (same format as concurrent batch)
        for i, result in enumerate(batch_results):
            # Unpack result - could be (text, reason, usage) or (text, reason, usage, error)
            if len(result) == 4:
                generated_text, finish_reason, token_usage, error = result
            else:
                generated_text, finish_reason, token_usage = result
                error = None

            sample = dataset[i]
            prompt = sample["query"]
            ground_truth = get_ground_truth(sample)

            # Handle errors
            if error:
                api_errors += 1
                print(f"\nError on sample {i}: {error}")
                predictions.append(build_prediction_dict(
                    sample, beginning + i, prompt, ground_truth,
                    None, f"ERROR: {error}", error=error
                ))
                continue

            # Handle None or empty content
            if generated_text is None or generated_text == "":
                empty_content_count += 1
                print(f"\n WARNING - Sample {i}: Empty content detected!")
                print(f"   Finish reason: {finish_reason}")
                generated_text = generated_text or ""

            # Track blocked content
            if finish_reason in ['SAFETY', 'RECITATION', 'OTHER', 'EMPTY_RESPONSE']:
                length_cutoff_count += 1
                print(f"\n Sample {i}: Content blocked - {finish_reason}")

            predicted = get_prediction(generated_text)

            if predicted is None:
                no_answer_count += 1
            if check_correctness(predicted, ground_truth):
                correct += 1

            predictions.append(build_prediction_dict(
                sample, beginning + i, prompt, ground_truth,
                predicted, generated_text, finish_reason, token_usage
            ))

    # Use official Batch API for OpenAI
    elif model_type == "openai":
        if not api_key:
            raise ValueError("API key required for official Batch API")

        print(f"Using official OpenAI Batch API (50% cost reduction, may take up to 24 hours)...")
        prompts = [sample["query"] for sample in dataset]

        # Run official batch processing
        batch_results = batch_call_openai_official(
            model_name=model_name,
            prompts=prompts,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            poll_interval=batch_poll_interval
        )

        # Process batch results (same format as concurrent batch)
        for i, result in enumerate(batch_results):
            # Unpack result - could be (text, reason, usage) or (text, reason, usage, error)
            if len(result) == 4:
                generated_text, finish_reason, token_usage, error = result
            else:
                generated_text, finish_reason, token_usage = result
                error = None

            sample = dataset[i]
            prompt = sample["query"]
            ground_truth = get_ground_truth(sample)

            # Handle errors
            if error:
                api_errors += 1
                print(f"\nError on sample {i}: {error}")
                predictions.append(build_prediction_dict(
                    sample, beginning + i, prompt, ground_truth,
                    None, f"ERROR: {error}", error=error
                ))
                continue

            # Handle None or empty content
            if generated_text is None or generated_text == "":
                empty_content_count += 1
                print(f"\n WARNING - Sample {i}: Empty content detected!")
                print(f"   Finish reason: {finish_reason}")
                generated_text = generated_text or ""

            # Track token limit cutoffs
            if finish_reason == "length":
                length_cutoff_count += 1
                print(f"\n Sample {i}: Response cut off due to token limit")

            predicted = get_prediction(generated_text)

            if predicted is None:
                no_answer_count += 1
            if check_correctness(predicted, ground_truth):
                correct += 1

            predictions.append(build_prediction_dict(
                sample, beginning + i, prompt, ground_truth,
                predicted, generated_text, finish_reason, token_usage
            ))

    # Use concurrent async batch processing for Kimi
    elif model_type == "kimi":
        if not api_key:
            raise ValueError("API key required for Kimi")
        print(f"Using concurrent batch processing with max {max_concurrent} concurrent requests for Kimi...")
        prompts = [sample["query"] for sample in dataset]
        kimi_client = AsyncOpenAI(api_key=api_key, base_url=KIMI_BASE_URL, timeout=600.0)

        # Run async batch processing
        batch_results = asyncio.run(batch_call_kimi(kimi_client, model_name, prompts, max_concurrent))

        # Process batch results
        for i, (generated_text, finish_reason, token_usage, *error) in enumerate(batch_results):
            sample = dataset[i]
            prompt = sample["query"]
            ground_truth = get_ground_truth(sample)

            # Handle errors
            if error:
                api_errors += 1
                print(f"\nError on sample {i}: {error[0]}")
                predictions.append(build_prediction_dict(
                    sample, beginning + i, prompt, ground_truth,
                    None, f"ERROR: {error[0]}", error=error[0]
                ))
                continue

            # Handle None or empty content
            if generated_text is None or generated_text == "":
                empty_content_count += 1
                print(f"\n WARNING - Sample {i}: Empty content detected!")
                print(f"   Finish reason: {finish_reason}")
                generated_text = generated_text or ""

            # Track token limit cutoffs
            if finish_reason == "length":
                length_cutoff_count += 1
                print(f"\n Sample {i}: Response cut off due to token limit")

            predicted = get_prediction(generated_text)

            if predicted is None:
                no_answer_count += 1
            if check_correctness(predicted, ground_truth):
                correct += 1

            predictions.append(build_prediction_dict(
                sample, beginning + i, prompt, ground_truth,
                predicted, generated_text, finish_reason, token_usage
            ))

    # Use concurrent async batch processing for GLM
    elif model_type == "glm":
        if not api_key:
            raise ValueError("API key required for GLM")
        print(f"Using concurrent batch processing with max {max_concurrent} concurrent requests for GLM...")
        prompts = [sample["query"] for sample in dataset]
        glm_client = AsyncOpenAI(api_key=api_key, base_url=GLM_BASE_URL)

        # Run async batch processing
        batch_results = asyncio.run(batch_call_glm(glm_client, model_name, prompts, max_concurrent))

        # Process batch results
        for i, (generated_text, finish_reason, token_usage, *error) in enumerate(batch_results):
            sample = dataset[i]
            prompt = sample["query"]
            ground_truth = get_ground_truth(sample)

            # Handle errors
            if error:
                api_errors += 1
                print(f"\nError on sample {i}: {error[0]}")
                predictions.append(build_prediction_dict(
                    sample, beginning + i, prompt, ground_truth,
                    None, f"ERROR: {error[0]}", error=error[0]
                ))
                continue

            # Handle None or empty content
            if generated_text is None or generated_text == "":
                empty_content_count += 1
                print(f"\n WARNING - Sample {i}: Empty content detected!")
                print(f"   Finish reason: {finish_reason}")
                generated_text = generated_text or ""

            # Track token limit cutoffs
            if finish_reason == "length":
                length_cutoff_count += 1
                print(f"\n Sample {i}: Response cut off due to token limit")

            predicted = get_prediction(generated_text)

            if predicted is None:
                no_answer_count += 1
            if check_correctness(predicted, ground_truth):
                correct += 1

            predictions.append(build_prediction_dict(
                sample, beginning + i, prompt, ground_truth,
                predicted, generated_text, finish_reason, token_usage
            ))

    # Use concurrent async batch processing for GPT-OSS (via Together.ai)
    elif model_type == "gpt-oss":
        if not api_key:
            raise ValueError("API key required for GPT-OSS")
        print(f"Using concurrent batch processing with max {max_concurrent} concurrent requests for GPT-OSS...")
        prompts = [sample["query"] for sample in dataset]
        gptoss_client = AsyncOpenAI(api_key=api_key, base_url=GLM_BASE_URL)

        # Run async batch processing (reuse GLM functions since both use Together.ai)
        batch_results = asyncio.run(batch_call_glm(gptoss_client, model_name, prompts, max_concurrent))

        # Process batch results
        for i, (generated_text, finish_reason, token_usage, *error) in enumerate(batch_results):
            sample = dataset[i]
            prompt = sample["query"]
            ground_truth = get_ground_truth(sample)

            # Handle errors
            if error:
                api_errors += 1
                print(f"\nError on sample {i}: {error[0]}")
                predictions.append(build_prediction_dict(
                    sample, beginning + i, prompt, ground_truth,
                    None, f"ERROR: {error[0]}", error=error[0]
                ))
                continue

            # Handle None or empty content
            if generated_text is None or generated_text == "":
                empty_content_count += 1
                print(f"\n WARNING - Sample {i}: Empty content detected!")
                print(f"   Finish reason: {finish_reason}")
                generated_text = generated_text or ""

            # Track token limit cutoffs
            if finish_reason == "length":
                length_cutoff_count += 1
                print(f"\n Sample {i}: Response cut off due to token limit")

            predicted = get_prediction(generated_text)

            if predicted is None:
                no_answer_count += 1
            if check_correctness(predicted, ground_truth):
                correct += 1

            predictions.append(build_prediction_dict(
                sample, beginning + i, prompt, ground_truth,
                predicted, generated_text, finish_reason, token_usage
            ))

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0

    # Calculate F1 score for binary labels
    tp = sum(1 for p in predictions if p.get('predicted_answer') == True and p.get('ground_truth_label') == True)
    fp = sum(1 for p in predictions if p.get('predicted_answer') == True and p.get('ground_truth_label') == False)
    fn = sum(1 for p in predictions if p.get('predicted_answer') == False and p.get('ground_truth_label') == True)
    tn = sum(1 for p in predictions if p.get('predicted_answer') == False and p.get('ground_truth_label') == False)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        'beginning': beginning,
        'model': model_name,
        'model_type': model_type,
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'no_answer_count': no_answer_count,
        'api_errors': api_errors,
        'empty_content_count': empty_content_count,
        'length_cutoff_count': length_cutoff_count,
    }

    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {model_name}")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"F1 Score: {f1:.2%} (P={precision:.2%}, R={recall:.2%})")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"\nNo Answer Count: {no_answer_count}")
    print(f"API Errors: {api_errors}")
    print(f"Empty Content Responses: {empty_content_count}")
    print(f"Token Limit Cutoffs: {length_cutoff_count}")
    print(f"{'='*80}\n")


    # Save all predictions if output file is specified
    output_data = {
            'metadata': results,
            'predictions': predictions
        }
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    print(f"All predictions saved to: {output_file}")

    return results


def infer_model_type(model_name: str) -> str:
    """Infer model_type from model_name.

    Returns one of: "openai", "gemini", "kimi", "glm", "gpt-oss"
    """
    name_lower = model_name.lower()

    # GPT-OSS (must check before openai since it contains 'gpt')
    if 'gpt-oss' in name_lower:
        return "gpt-oss"

    # OpenAI models: gpt-4*, gpt-5*, o1*, o3*
    if name_lower.startswith(('gpt-', 'o1', 'o3')):
        return "openai"

    # Gemini models
    if 'gemini' in name_lower:
        return "gemini"

    # Kimi models: moonshot-*, kimi-*, k2-*
    if name_lower.startswith(('moonshot', 'kimi', 'k2')):
        return "kimi"

    # GLM models: GLM-*, glm-*, zai-org/GLM-*
    if 'glm' in name_lower:
        return "glm"

    raise ValueError(f"Cannot infer model_type from model_name: {model_name}. "
                     "Please use a recognizable model name or specify model_type explicitly.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on causal reasoning dataset")
    parser.add_argument("model_name", type=str, help="Name of the model to evaluate")
    parser.add_argument("--max-samples", type=int, default=300, help="Maximum samples to evaluate")
    parser.add_argument("--beginning", type=int, default=0, help="Starting index in dataset")
    parser.add_argument("--reasoning-effort", type=str, default="high",
                        choices=["low", "medium", "high"], help="OpenAI reasoning effort level")
    parser.add_argument("--thinking-level", type=str, default="high",
                        help="Gemini thinking level (low/high for Gemini 3)")
    parser.add_argument("--max-concurrent", type=int, default=10,
                        help="Max concurrent requests for Kimi/GLM/GPT-OSS")
    parser.add_argument("--batch-poll-interval", type=int, default=60,
                        help="Polling interval for batch API (seconds)")
    parser.add_argument("--for-sft", action="store_true", help="Use SFT data path")
    parser.add_argument("--split", type=str, default="c2cpSplit1",
                        help="Dataset split to use (e.g., c2cpSplit1, c2cpSplit2, c2cpSplit3)")
    args = parser.parse_args()

    model_name = args.model_name
    model_type = infer_model_type(model_name)
    print(f"Inferred model_type: {model_type}")

    # Data path
    if args.for_sft:
        data_path = f"./data/processed/{args.split}/flash"
    else:
        data_path = f"./data/processed/{args.split}/else"

    # Output file for saving all predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_parts = data_path.rstrip("/").split("/")
    dataset_name = path_parts[-2] if len(path_parts) >= 2 else "unknown"

    # Create output directory with dataset and model_type subfolders
    output_dir = os.path.join("./evaluation_results", dataset_name, model_name)
    if model_type == "openai":
        model_prefix = "gpt" + model_name
    elif model_type == "kimi":
        model_prefix = "kimi" + model_name
    elif model_type == "glm":
        model_prefix = "glm" + model_name
    elif model_type == "gpt-oss":
        model_prefix = "gpt-oss" + model_name.replace("/", "_")
    else:  # gemini
        model_prefix = "gemini" + model_name
    output_file = os.path.join(output_dir, f"{model_prefix}_NEW_{model_name}_{timestamp}.json")

    # Get API key based on model type
    if model_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
    elif model_type == "kimi":
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("Please set MOONSHOT_API_KEY environment variable")
    elif model_type in ("glm", "gpt-oss"):
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("Please set TOGETHER_API_KEY environment variable")
    elif model_type == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Please set GOOGLE_API_KEY environment variable")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'openai', 'kimi', 'glm', 'gpt-oss', or 'gemini'")

    # Load dataset
    print(f"Loading dataset from {data_path}...")
    if args.for_sft:
        dataset = load_from_disk(data_path)['train']
    else:
        dataset = load_from_disk(data_path)['test']
    print(f"Dataset loaded: {len(dataset)} samples in data set")

    # Evaluate
    results = evaluate_model_on_dataset(
        model_name=model_name,
        dataset=dataset,
        max_samples=args.max_samples,
        output_file=output_file,
        api_key=api_key,
        model_type=model_type,
        reasoning_effort=args.reasoning_effort,
        thinking_level=args.thinking_level,
        beginning=args.beginning,
        max_concurrent=args.max_concurrent,
        batch_poll_interval=args.batch_poll_interval,
    )

    print("Evaluation complete!")
    print(f"Results summary saved to: {output_file}")


if __name__ == "__main__":
    main()
