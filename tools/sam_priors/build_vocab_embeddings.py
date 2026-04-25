"""
Build CLIP text embeddings for ImageNet vocabulary filtered by COCO base classes.

Saves pretrained/vocab_embeddings.pt with keys:
  names      : list[str]  — filtered vocab entries
  embeddings : (M, 1024) float32 tensor, L2-normalized
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.clip.clip import _MODELS, _download, tokenize

# ---------------------------------------------------------------------------
# Standard ImageNet 1000 class names (ILSVRC-2012 human-readable labels)
# ---------------------------------------------------------------------------
IMAGENET_CLASSES = [
    # Fish
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
    "electric ray", "stingray",
    # Birds
    "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee",
    "water ouzel", "kite", "bald eagle", "vulture", "great grey owl",
    # Amphibians / reptiles
    "fire salamander", "smooth newt", "eft", "spotted salamander", "axolotl",
    "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle",
    "leatherback sea turtle", "mud turtle", "terrapin", "box turtle",
    "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard",
    "agama", "frilled-neck lizard", "alligator lizard", "Gila monster",
    "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
    "American alligator", "triceratops",
    # Snakes
    "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake",
    "kingsnake", "garter snake", "water snake", "vine snake", "night snake",
    "boa constrictor", "African rock python", "Indian cobra", "green mamba",
    "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
    "sidewinder rattlesnake",
    # Invertebrates
    "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider",
    "European garden spider", "southern black widow", "tarantula", "wolf spider",
    "tick", "centipede",
    # More birds
    "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peacock",
    "quail", "partridge", "African grey parrot", "macaw", "sulphur-crested cockatoo",
    "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar",
    "toucan", "mallard", "red-breasted merganser", "puffin", "black swan",
    "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
    "great egret", "bittern", "crane", "limpkin", "common gallinule",
    "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank",
    "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross",
    # Marine mammals
    "grey whale", "killer whale", "dugong", "sea lion",
    # Dog breeds (120 classes in ImageNet)
    "Chihuahua", "Japanese chin", "Maltese", "Pekingese", "Shih Tzu",
    "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback",
    "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound",
    "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
    "English Foxhound", "Redbone Coonhound", "Borzoi", "Irish Wolfhound",
    "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound",
    "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner",
    "Staffordshire Bull Terrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier",
    "Wirehaired Fox Terrier", "Lakeland Terrier", "Sealyham Terrier",
    "Airedale Terrier", "Cairn Terrier", "Australian Terrier",
    "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
    "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier",
    "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier",
    "West Highland White Terrier", "Lhasa Apso", "Flat-coated Retriever",
    "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever",
    "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
    "English Setter", "Irish Setter", "Gordon Setter", "Brittany",
    "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel",
    "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel",
    "Kuvasz", "Schipperke", "Groenendael", "Malinois", "Briard",
    "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog",
    "Rough Collie", "Border Collie", "Bouvier des Flandres", "Rottweiler",
    "German Shepherd", "Dobermann", "Miniature Pinscher",
    "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Mountain Dog", "Boxer",
    "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane",
    "Saint Bernard", "Siberian Husky", "Alaskan Malamute", "Dalmatian",
    "Affenpinscher", "Basenji", "Pug", "Leonberger", "Newfoundland",
    "Great Pyrenees", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond",
    "Brussels Griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi",
    "Toy Poodle", "Miniature Poodle", "Standard Poodle",
    "Mexican Hairless Dog",
    # Wild canids / felids
    "grey wolf", "coyote", "dingo", "dhole", "African wild dog",
    "hyena", "red fox", "kit fox", "Arctic fox", "grey fox",
    "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau",
    "cougar", "lynx", "leopard", "snow leopard", "jaguar",
    # More mammals
    "brown bear", "American black bear", "polar bear", "sloth bear",
    "mongoose", "meerkat",
    # Insects
    "tiger beetle", "ladybug", "ground beetle", "long-horned beetle",
    "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil",
    "fly", "bee", "ant", "grasshopper", "cricket", "walking stick insect",
    "cockroach", "mantis", "cicada", "leafhopper", "lacewing",
    "dragonfly", "damselfly", "admiral butterfly", "ringlet butterfly",
    "monarch butterfly", "cabbage butterfly", "sulphur butterfly",
    "lycaenid butterfly",
    # Marine invertebrates
    "starfish", "sea urchin", "sea cucumber",
    # Small mammals
    "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine",
    "fox squirrel", "marmot", "beaver", "guinea pig",
    # Ungulates
    "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
    "ram", "bighorn sheep", "Alpine ibex", "hartebeest", "impala", "gazelle",
    "dromedary camel", "llama",
    # Mustelids and other small mammals
    "weasel", "mink", "European polecat", "black-footed ferret", "otter",
    "skunk", "badger", "armadillo", "three-toed sloth",
    # Primates
    "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon",
    "patas monkey", "baboon", "macaque", "langur",
    "black-and-white colobus", "proboscis monkey", "marmoset",
    "white-headed capuchin", "howler monkey", "spider monkey",
    "squirrel monkey", "ring-tailed lemur", "indri",
    # Pachyderms, etc.
    "Asian elephant", "African bush elephant", "lesser panda", "giant panda",
    # Farm animals
    "sorrel horse", "zebra animal",
    # Vehicles - ground
    "garbage truck", "pickup truck", "ambulance", "fire engine",
    "police van", "jeep", "tractor", "trailer truck", "moving van",
    "recreational vehicle", "streetcar", "trolleybus", "school bus",
    "minibus", "tank", "half-track", "motor scooter", "mountain bike",
    "moped", "go-kart", "snowmobile", "snowplow", "tow truck",
    "forklift", "steam roller", "bulldozer",
    # Vehicles - air
    "space shuttle", "airliner", "airship", "parachute", "warplane",
    # Vehicles - water
    "liner ship", "gondola boat", "submarine", "speedboat", "canoe",
    "catamaran", "trimaran", "yawl", "schooner", "container ship",
    "amphibious vehicle", "bobsled", "dog sled",
    # Rail
    "freight car", "passenger car", "electric locomotive",
    # Electronics and technology
    "barometer", "oscilloscope", "magnetic compass", "sundial",
    "digital watch", "analog clock", "digital clock", "wall clock",
    "hourglass", "stopwatch", "abacus", "cash machine",
    "hand-held computer", "laptop computer", "desktop computer",
    "computer monitor", "computer keyboard", "joystick", "typewriter keyboard",
    "modem", "radio", "tape player", "CD player", "television",
    "entertainment center", "projector", "home theater", "microphone",
    "loudspeaker", "power drill", "hand blower", "vacuum cleaner",
    "washing machine", "dishwasher", "refrigerator", "Dutch oven",
    "waffle iron", "frying pan", "electric fan",
    "stove", "toaster", "coffeepot", "espresso maker",
    # Food and drink
    "red wine", "coffee cup", "mixing bowl", "soup bowl",
    "pitcher", "beer bottle", "beer glass", "goblet", "wine bottle",
    "milk can", "bucket", "barrel", "pot",
    # Kitchen utensils
    "ladle", "wooden spoon", "spatula", "measuring cup", "saltshaker",
    "can opener", "corkscrew", "bottle cap",
    # Medical
    "medicine chest", "pill bottle", "syringe", "stethoscope",
    "Petri dish", "bandage", "crutch", "stretcher", "neck brace",
    # Office and stationery
    "slide rule", "pencil box", "pencil sharpener", "printer",
    "photocopier", "paper towel", "envelope", "file cabinet",
    # Outdoor structures
    "mailbox", "parking meter", "pay phone", "vending machine",
    "traffic cone", "guardrail", "manhole cover",
    # Weapons and tools
    "rifle", "revolver", "assault rifle", "holster", "missile", "cannon",
    "axe", "cleaver", "hatchet", "hammer", "saw", "wrench", "pliers",
    "screwdriver", "shovel", "broom",
    # Musical instruments
    "electric guitar", "banjo", "violin", "cello", "upright piano",
    "grand piano", "organ", "accordion", "harmonica", "ocarina",
    "panpipe", "flute", "oboe", "bassoon", "French horn", "trombone",
    "cornet", "marimba", "gong", "steel drum", "drum",
    "xylophone", "maraca", "chime", "harp", "sitar",
    # Architecture / structures
    "breakwater", "pier", "dock", "dam", "suspension bridge",
    "steel arch bridge", "triumphal arch", "castle", "palace",
    "monastery", "church", "mosque", "stupa", "dome",
    "cliff dwelling", "yurt", "greenhouse", "boathouse", "lumbermill",
    "bakery", "barbershop", "grocery store", "bookstore", "restaurant",
    "cinema", "stage", "ballpark", "planetarium", "beacon", "flagpole",
    "totem pole", "bell tower", "tile roof", "thatch roof",
    "chain-link fence", "picket fence", "stone wall", "iron fence",
    # Sports equipment
    "barbell", "dumbbell", "balance beam", "horizontal bar",
    "parallel bars", "vault", "ski", "snowboard", "surfboard",
    "volleyball", "rugby ball", "golf ball", "ping-pong ball",
    "croquet ball", "puck", "tennis racket", "fishing rod", "paddle",
    "golf club", "bow", "crossbow", "billiard table",
    "slot machine", "jigsaw puzzle", "teddy bear",
    # Clothing and accessories
    "bathing cap", "bikini", "swimming trunks", "jersey", "sweatshirt",
    "gown", "kimono", "sarong", "hoopskirt", "miniskirt", "trench coat",
    "fur coat", "lab coat", "cloak", "poncho", "bonnet", "cowboy hat",
    "sombrero", "crash helmet", "football helmet", "ski mask", "gasmask",
    "bra", "sock", "mitten", "knee pad", "breastplate", "cuirass",
    "shield", "chain mail", "bulletproof vest", "backpack", "purse",
    "wallet", "diaper", "apron", "bib",
    # Beauty and personal care
    "face powder", "lipstick", "lotion", "perfume", "wig", "sunscreen",
    "sunglass",
    # Optical instruments
    "binoculars", "loupe",
    # Furniture
    "rocking chair", "studio couch", "folding chair", "park bench",
    "cradle", "crib", "bookcase", "china cabinet", "chiffonier",
    "wardrobe", "chest", "desk", "dining table", "dresser",
    "shower curtain", "pillow", "quilt", "sleeping bag",
    # Miscellaneous objects
    "padlock", "combination lock", "safety pin", "buckle", "coil",
    "rubber eraser", "paintbrush", "plunger", "fire extinguisher",
    "funicular", "crane machine", "oil rig", "drilling platform",
    "satellite dish", "streetlight", "water tower", "silo",
    "swimming pool", "fountain", "radiator", "fireplace", "chimney",
    "smoke stack",
    # Natural scenes and objects
    "coral reef", "geyser", "lakeshore", "promontory", "sandbar",
    "seashore", "valley", "volcano",
    # People (occupational)
    "baseball player", "bridegroom", "scuba diver",
    # Plants
    "rapeseed", "daisy", "yellow lady slipper", "corn plant",
    "acorn", "hip berry", "buckeye nut",
    # Fungi
    "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
    "earthstar mushroom", "hen-of-the-woods mushroom", "bolete mushroom",
    # Miscellaneous final
    "comic book", "crossword puzzle", "street sign",
    "book jacket", "menu",
    "guacamole", "consomme", "hot pot", "trifle", "ice cream",
    "ice lolly", "French loaf", "bagel", "pretzel",
    "mashed potato", "head cabbage",
    "broccoli vegetable", "cauliflower", "zucchini", "spaghetti squash",
    "acorn squash", "butternut squash", "cucumber", "artichoke",
    "bell pepper", "cardoon", "mushroom", "Granny Smith apple",
    "strawberry", "orange fruit", "lemon", "fig", "pineapple",
    "banana fruit", "jackfruit", "cherimoya", "pomegranate",
    "carbonara", "chocolate sauce", "dough", "meat loaf",
    "potpie", "burrito", "espresso coffee", "eggnog",
    "alp mountain", "bubble", "cliff", "coral", "geyser rock",
    "rapeseed field", "hay bale",
]

# Deduplicate preserving order
_seen = set()
_deduped = []
for x in IMAGENET_CLASSES:
    if x.lower() not in _seen:
        _seen.add(x.lower())
        _deduped.append(x)
IMAGENET_CLASSES = _deduped


def _load_clip_text_encoder(model_name: str = "RN50", device: str = "cpu"):
    """Return (encode_text_fn, tokenize_fn) using the project CLIP text encoder."""
    from models.backbone.base import Classifier
    classifier = Classifier(model_name=f"CLIP_{model_name}", token_len=77)
    classifier.eval()
    classifier.to(device)
    return classifier


@torch.no_grad()
def compute_embeddings(names, classifier, device, batch=256):
    all_embs = []
    for i in tqdm(range(0, len(names), batch), desc="encoding text"):
        batch_names = names[i : i + batch]
        texts = [f"a photo of a {n}" for n in batch_names]
        tokens = tokenize(texts, context_length=77, truncate=True).to(device)
        embs = classifier.encode_text(tokens)
        embs = F.normalize(embs.float(), dim=-1)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ann", default="data/Annotations/instances_train2017_base.json",
                        help="COCO base annotation file to extract base class names from")
    parser.add_argument("--out", default="pretrained/vocab_embeddings.pt",
                        help="Output path for vocab embeddings")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load COCO base class names and build filter set
    base_names = set()
    if Path(args.base_ann).exists():
        with open(args.base_ann) as f:
            cats = json.load(f)["categories"]
        base_names = {c["name"].lower() for c in cats}
        print(f"Loaded {len(base_names)} COCO base class names to exclude.")
    else:
        print(f"Warning: {args.base_ann} not found; skipping COCO base filtering.")

    # Filter vocab
    vocab = [n for n in IMAGENET_CLASSES if n.lower() not in base_names]
    print(f"Vocabulary after filtering: {len(vocab)} entries (from {len(IMAGENET_CLASSES)})")

    # Load CLIP text encoder
    print("Loading CLIP RN50 text encoder…")
    classifier = _load_clip_text_encoder("RN50", args.device)

    # Compute embeddings
    embs = compute_embeddings(vocab, classifier, args.device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"names": vocab, "embeddings": embs}, out_path)
    print(f"Saved {len(vocab)} embeddings ({embs.shape}) → {out_path}")


if __name__ == "__main__":
    main()
