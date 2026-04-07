#!/usr/bin/env python3
"""
Test script to export metadata from XML files to individual JSON files.
Each song gets its own JSON file in dataset/metadata/
"""

import sys
sys.path.append('/Users/david/ANIMA_Data_Formation/src')

from utils import export_all_metadata_from_xml, get_metadata
import json

# Test single file first
print("Testing single file metadata extraction...")
test_xml = "../dataset/iRealXML/A Night In Tunisia.xml"
metadata = get_metadata(test_xml)

print("\nExtracted metadata:")
for key, value in metadata.items():
    print(f"  {key}: {value}")

# Export all metadata
print("\n" + "="*60)
print("Exporting all metadata to JSON files...")
print("="*60 + "\n")

exported_files = export_all_metadata_from_xml(
    xml_dir="../dataset/iRealXML",
    output_dir="../dataset/metadata"
)

print(f"\n✅ Successfully exported {len(exported_files)} metadata files!")

# Show a few examples
print("\nFirst 5 exported files:")
for f in exported_files[:5]:
    print(f"  - {f}")
