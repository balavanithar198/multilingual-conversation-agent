"""
Setup Checker — Run this first to verify everything is configured.
Usage: python check_setup.py
"""

import os

def main():
    print()
    print("=" * 50)
    print("  TANGLISH AGENT - SETUP CHECK")
    print("=" * 50)
    print()

    # Check .env file
    if not os.path.exists(".env"):
        print("  ❌ .env file NOT FOUND")
        print("  → Copy .env.example to .env and add your keys")
        print()
        return

    print("  ✅ .env file exists")

    # Load and check keys
    from dotenv import load_dotenv
    load_dotenv()

    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "LIVEKIT_URL": os.getenv("LIVEKIT_URL"),
        "LIVEKIT_API_KEY": os.getenv("LIVEKIT_API_KEY"),
        "LIVEKIT_API_SECRET": os.getenv("LIVEKIT_API_SECRET"),
    }

    all_good = True
    for key, value in keys.items():
        if not value or "your" in value.lower() or "xxxx" in value.lower() or "REPLACE" in value:
            print(f"  ❌ {key}: NOT SET (still has placeholder)")
            all_good = False
        else:
            masked = value[:12] + "..."
            print(f"  ✅ {key}: {masked}")

    # Check packages
    print()
    try:
        import livekit.agents
        print("  ✅ livekit-agents installed")
    except ImportError:
        print("  ❌ livekit-agents NOT INSTALLED")
        all_good = False

    try:
        import dotenv
        print("  ✅ python-dotenv installed")
    except ImportError:
        print("  ❌ python-dotenv NOT INSTALLED")
        all_good = False

    # Result
    print()
    print("=" * 50)
    if all_good:
        print("  ✅ ALL GOOD! Run the agent with:")
        print("     python agent.py dev")
    else:
        print("  ❌ FIX THE ISSUES ABOVE, THEN RUN:")
        print('     pip install "livekit-agents[openai]~=1.5" python-dotenv')
        print("     python agent.py dev")
    print("=" * 50)
    print()


if __name__ == "__main__":
    main()