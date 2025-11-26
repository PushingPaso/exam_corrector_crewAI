import sys
import os
import time
from exam.rag import sqlite_vector_store, all_slides, FILE_DB


def print_separator():
    print("-" * 60)


def recreate_database():
    if os.path.exists(FILE_DB):
        try:
            os.remove(FILE_DB)
            print(f"Deleted existing database: {FILE_DB}")
        except OSError as e:
            print(f"Error deleting database: {e}")
            return

    print(f"Creating new database at: {FILE_DB}")

    try:
        vstore = sqlite_vector_store(table_name="se_slides")
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    slides = list(all_slides())
    total_slides = len(slides)
    print(f"Found {total_slides} slides to process.")

    count = 0
    skipped = 0
    start_time = time.time()

    for s in slides:
        try:
            success = vstore.add_slide(s)
            if success:
                count += 1
                if count % 50 == 0:
                    print(f"Processed {count}/{total_slides} slides...")
            else:
                skipped += 1
        except Exception as e:
            print(f"Error processing slide {s.index} ({s.source}): {e}")

    vstore.close()

    elapsed = time.time() - start_time
    print_separator()
    print(f"Database generation complete.")
    print(f"Total added: {count}")
    print(f"Total skipped: {skipped}")
    print(f"Time taken: {elapsed:.2f}s")


def interactive_search():
    if not os.path.exists(FILE_DB):
        print(f"ERROR: Database {FILE_DB} not found.")
        print("Run 'python main.py --fill' to generate it.")
        return

    print(f"Loading database from: {FILE_DB}")

    try:
        vstore = sqlite_vector_store(table_name="se_slides")
    except Exception as e:
        print(f"Connection error: {e}")
        return

    print("\nSystem ready. Type your query (or 'exit' to quit).")
    print_separator()

    while True:
        try:
            query = input("\nQuery: ").strip()

            if query.lower() in ["exit", "quit"]:
                print("Exiting...")
                break

            if not query:
                continue

            print("Searching...")

            results = vstore.search(query, k=3)

            if not results:
                print("No results found.")
                continue

            print(f"\nFound {len(results)} results:\n")

            for i, res in enumerate(results, 1):
                dist = res['distance']
                source_clean = res['source'].replace("\\", "/")

                print(f"#{i} [Distance: {dist:.4f}] {source_clean} (Slide {res['index']})")
                print("..." + res['content'].replace("\n", " ")[:200] + "...")
                print_separator()

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nSearch error: {e}")

    vstore.close()


def main():
    if "--fill" in sys.argv:
        recreate_database()
    else:
        interactive_search()


if __name__ == "__main__":
    main()