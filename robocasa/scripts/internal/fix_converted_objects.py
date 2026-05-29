import os
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse


def fix_texture_paths_and_duplicates(xml_path):
    """
    Remove duplicate texture entries and fix absolute texture paths in an MJCF file.

    Args:
        xml_path (Path): Path to the XML file to process

    Returns:
        bool: True if changes were made, False otherwise
    """
    try:
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find the asset element
        asset = root.find("asset")
        if asset is None:
            return False

        # Track unique textures by their name and file attributes
        seen_textures = {}
        textures_to_remove = []
        made_changes = False

        # Process all textures
        for texture in asset.findall("texture"):
            name = texture.get("name")
            file_path = texture.get("file")

            if name is None or file_path is None:
                continue

            # Check if texture file exists and is accessible
            texture_path = (Path(xml_path).parent / file_path).resolve()

            # check that the texture path is "under" the XML file's parent directory

            # if texture file does not exist or is not in the same directory as the XML file, replace with fallback texture
            if (not texture_path.is_file()) or (
                not Path(xml_path).parent.resolve() in texture_path.parents
            ):
                # Replace with fallback texture
                new_texture = ET.Element(
                    "texture",
                    {
                        "builtin": "flat",
                        "width": "1",
                        "height": "1",
                        "name": name,
                        "type": "2d",
                    },
                )
                texture_idx = list(asset).index(texture)
                asset.remove(texture)
                asset.insert(texture_idx, new_texture)
                made_changes = True
                texture = new_texture

            # Handle duplicates
            key = (name, texture.get("file") if texture.get("file") else "builtin")
            if key in seen_textures:
                textures_to_remove.append(texture)
                made_changes = True
            else:
                seen_textures[key] = texture

        # Remove duplicate textures
        for texture in textures_to_remove:
            asset.remove(texture)

        # If any changes were made, save the file
        if made_changes:
            # Write the modified XML with proper formatting
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            return True

        return False

    except ET.ParseError as e:
        print(f"Error parsing {xml_path}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return False


def process_all_models(obj_registries):
    """
    Process all model.xml files in the objects directory structure.

    Args:
        obj_registries (list[str]): Object registries to process
    """
    # Get path relative to this script
    script_dir = Path(__file__).resolve().parent
    objects_path = script_dir.parent.parent / "models" / "assets" / "objects"

    print(f"objects_path: {objects_path}")

    if not objects_path.exists():
        print(f"Objects directory not found: {objects_path}")
        return

    modified_files = 0
    processed_files = 0

    # Walk through the directory structure
    for registry_dir in objects_path.iterdir():
        if not registry_dir.is_dir():
            continue

        if registry_dir.name not in obj_registries:
            continue

        for object_dir in registry_dir.iterdir():
            if not object_dir.is_dir():
                continue

            for variant_dir in object_dir.iterdir():
                if not variant_dir.is_dir():
                    continue

                model_file = variant_dir / "model.xml"
                if model_file.exists():
                    processed_files += 1
                    if fix_texture_paths_and_duplicates(model_file):
                        modified_files += 1
                        print(f"Modified: {model_file.relative_to(objects_path)}")

    print(f"\nProcessing complete:")
    print(f"Total files processed: {processed_files}")
    print(f"Files modified: {modified_files}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix texture paths and remove duplicates from MJCF files"
    )
    parser.add_argument(
        "--obj-registries",
        type=str,
        nargs="+",
        help="Object registries to process. Choose among [infinigen, aigen, objaverse, sketchfab]",
    )

    args = parser.parse_args()
    process_all_models(args.obj_registries)


if __name__ == "__main__":
    main()
