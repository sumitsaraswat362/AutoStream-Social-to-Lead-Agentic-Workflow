"""
AutoStream Social-to-Lead Agent — Entry Point

A conversational AI agent that:
  1. Understands user intent (greeting / product inquiry / high intent)
  2. Answers product questions from a local knowledge base (RAG)
  3. Detects when a user is ready to buy
  4. Progressively collects lead details (name → email → platform)
  5. Fires a mock lead capture function

Usage:
    python main.py
"""

import sys
from agent import AutoStreamAgent


# ═════════════════════════════════════════════════════════════════════════════
# Terminal Colors (ANSI escape codes)
# ═════════════════════════════════════════════════════════════════════════════

class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# ═════════════════════════════════════════════════════════════════════════════
# Startup Banner
# ═════════════════════════════════════════════════════════════════════════════

BANNER = f"""
{Colors.CYAN}{'═' * 60}
   █████╗ ██╗   ██╗████████╗ ██████╗ 
  ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗
  ███████║██║   ██║   ██║   ██║   ██║
  ██╔══██║██║   ██║   ██║   ██║   ██║
  ██║  ██║╚██████╔╝   ██║   ╚██████╔╝
  ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ 
      {Colors.MAGENTA}S T R E A M{Colors.CYAN}
{'═' * 60}{Colors.RESET}
  {Colors.DIM}AI-Powered Video Editing for Content Creators{Colors.RESET}
  {Colors.DIM}Social-to-Lead Conversational Agent v1.0{Colors.RESET}
{Colors.CYAN}{'═' * 60}{Colors.RESET}
  {Colors.YELLOW}Type your message and press Enter.
  Type 'quit' or 'exit' to end the conversation.{Colors.RESET}
{Colors.CYAN}{'═' * 60}{Colors.RESET}
"""


# ═════════════════════════════════════════════════════════════════════════════
# Main Loop
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """Run the interactive conversational agent loop."""
    print(BANNER)

    # Initialize the agent
    agent = AutoStreamAgent()
    print(f"  {Colors.DIM}Agent initialized. Ready to chat!{Colors.RESET}\n")

    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.GREEN}{Colors.BOLD}You: {Colors.RESET}")

            # Handle exit commands
            if user_input.strip().lower() in ("quit", "exit", "q", "bye"):
                print(f"\n{Colors.CYAN}Agent: Thanks for chatting! Have a great day! 👋{Colors.RESET}\n")
                break

            # Skip empty inputs
            if not user_input.strip():
                continue

            # Run the agent and display response
            response = agent.run(user_input)
            print(f"\n{Colors.CYAN}{Colors.BOLD}Agent:{Colors.RESET} {response}\n")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}Agent: Goodbye! 👋{Colors.RESET}\n")
            break
        except Exception as e:
            print(f"\n{Colors.YELLOW}[Error] {e}{Colors.RESET}")
            print(f"{Colors.DIM}Please try again or type 'quit' to exit.{Colors.RESET}\n")


if __name__ == "__main__":
    main()
