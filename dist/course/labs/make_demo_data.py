"""Create original synthetic practice documents. No downloads or model calls."""
import argparse
import itertools
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    places = ["garden", "workshop", "library", "kitchen", "studio", "station"]
    people = ["Mira", "Jon", "Asha", "Leo", "Nora", "Sam"]
    objects = ["blue notebook", "small clock", "paper map", "red box", "wooden chair", "green bag"]
    docs = []
    for person, place, obj in itertools.product(people, places, objects):
        docs.append(f"{person} visited the {place} in the morning. There was a {obj} near the window.\n"
                    f"The room was quiet. {person} looked at the {obj} and wrote a note about the visit.\n"
                    f"In the afternoon, {person} returned to the {place}. The light had changed, but the {obj} was still there.")
    random.Random(17).shuffle(docs)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, subset in (("train", docs[:180]), ("validation", docs[180:])):
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for index, text in enumerate(subset):
                handle.write(json.dumps({"id": f"{name}-{index:03d}", "text": text}, ensure_ascii=False) + "\n")
    print(f"Created 180 training and 36 validation documents in {args.out}.")
    print("Synthetic template data: suitable for learning mechanics, not measuring general language ability.")


if __name__ == "__main__":
    main()
