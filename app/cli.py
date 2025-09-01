import sys
from app.rag import ask

def main():
    q = " ".join(sys.argv[1:]) or "How do we fight fairly when we both feel criticized?"
    out = ask(q)
    print("\n=== Answer ===\n")
    print(out["answer"])
    print("\n=== Citations ===")
    for c in out["citations"]:
        print(f" - [{c['doc_id']}] {c['title']}")

if __name__ == "__main__":
    main()