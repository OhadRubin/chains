def calc_cost(metrics):
    # Calculate costs for Claude 3.5 Sonnet
    total_base_input_cost = sum(m["input_tokens"] for m in metrics) * (3 / 1_000_000)
    total_cache_writes_cost = sum(m["input_tokens_cache_create"] for m in metrics) * (
        3.75 / 1_000_000
    )
    total_cache_hits_cost = sum(m["input_tokens_cache_read"] for m in metrics) * (
        0.30 / 1_000_000
    )
    total_output_cost = sum(m["output_tokens"] for m in metrics) * (15 / 1_000_000)

    total_cost = (
        total_base_input_cost
        + total_cache_writes_cost
        + total_cache_hits_cost
        + total_output_cost
    )

    print(f"Base input cost: ${total_base_input_cost:.6f},")
    print(f"Cache writes cost: ${total_cache_writes_cost:.6f},")
    print(f"Cache hits cost: ${total_cache_hits_cost:.6f},")
    print(f"Output cost: ${total_output_cost:.6f},")
    print(f"Total cost: ${total_cost:.6f},")
    # Calculate costs for Claude 3.5 Sonnet without caching
    # All input tokens would be charged at base rate since no caching
    total_input_tokens = sum(
        m["input_tokens"]
        + m["input_tokens_cache_create"]
        + m["input_tokens_cache_read"]
        for m in metrics
    )
    no_cache_input_cost = total_input_tokens * (3 / 1_000_000)
    no_cache_output_cost = sum(m["output_tokens"] for m in metrics) * (15 / 1_000_000)

    no_cache_total = no_cache_input_cost + no_cache_output_cost

    print("\nWithout caching:")
    print(f"Input cost: ${no_cache_input_cost:.6f}")
    print(f"Output cost: ${no_cache_output_cost:.6f}")
    print(f"Total cost: ${no_cache_total:.6f}")
    print(f"\nSavings from caching: ${(no_cache_total - total_cost):.6f}")
