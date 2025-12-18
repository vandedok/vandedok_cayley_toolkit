import requests



def get_r2_bucket_usage_with_api(account_id: str, bucket_name: str, api_token: str) -> dict:
    """
    Get R2 bucket usage using Cloudflare API.
    Note: api size request has some delay, so the size returned may be not up-to-date.
    """
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/r2/buckets/{bucket_name}/usage"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers)

    json_data = response.json()
    result = json_data['result']

    # payload_size = result['payloadSize']
    # metadata_size = result['metadataSize']
    # object_count = result['objectCount']

    return result

