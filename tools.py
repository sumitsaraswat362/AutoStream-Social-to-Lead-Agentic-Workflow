"""
Tool definitions for the AutoStream Social-to-Lead agent.

Contains the mock lead capture function that simulates
sending qualified lead data to a CRM or backend system.

CRITICAL: mock_lead_capture must ONLY be called after all three
parameters (name, email, platform) have been collected from the user
through the conversation flow. Premature invocation is a failure condition.
"""


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulate capturing a qualified lead and sending it to a CRM.

    This function represents the final tool execution step in the
    Social-to-Lead workflow. It should only fire after the agent has
    progressively collected all three required fields through conversation.

    Args:
        name: The lead's full name
        email: The lead's email address
        platform: The content platform they use (YouTube, Instagram, TikTok, etc.)

    Returns:
        dict with status and the captured lead data
    """
    print(f"\n{'='*55}")
    print(f"  🎯 LEAD CAPTURED SUCCESSFULLY")
    print(f"{'='*55}")
    print(f"  Name:     {name}")
    print(f"  Email:    {email}")
    print(f"  Platform: {platform}")
    print(f"{'='*55}\n")

    return {
        "status": "success",
        "lead": {
            "name": name,
            "email": email,
            "platform": platform,
        },
    }
