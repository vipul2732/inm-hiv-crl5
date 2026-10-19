#!/bin/bash
# Download STRING, IntAct, and BioGRID databases

set -e  # Exit on error

DATABASE_DIR="$(cd "$(dirname "$0")/../../database" && pwd)"
cd "$DATABASE_DIR"

echo "=========================================="
echo "Downloading Protein Interaction Databases"
echo "=========================================="
echo ""

# Create a temporary directory for downloads
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# 1. Download STRING database (Human - 9606)
echo "1. Downloading STRING database (Human)..."
STRING_URL="https://stringdb-static.org/download/protein.links.detailed.v12.0/9606.protein.links.detailed.v12.0.txt.gz"
STRING_FILE="string_human_protein_links_detailed.txt.gz"

if command -v wget &> /dev/null; then
    wget -q --show-progress -O "$STRING_FILE" "$STRING_URL" || {
        echo "  Trying alternative STRING URL (v11.5)..."
        STRING_URL="https://stringdb-static.org/download/protein.links.detailed.v11.5/9606.protein.links.detailed.v11.5.txt.gz"
        wget -q --show-progress -O "$STRING_FILE" "$STRING_URL"
    }
elif command -v curl &> /dev/null; then
    curl -L --progress-bar -o "$STRING_FILE" "$STRING_URL" || {
        echo "  Trying alternative STRING URL (v11.5)..."
        STRING_URL="https://stringdb-static.org/download/protein.links.detailed.v11.5/9606.protein.links.detailed.v11.5.txt.gz"
        curl -L --progress-bar -o "$STRING_FILE" "$STRING_URL"
    }
else
    echo "Error: Neither wget nor curl is available. Please install one of them."
    exit 1
fi

# Decompress STRING file
if [ -f "$STRING_FILE" ]; then
    echo "  Decompressing STRING file..."
    gunzip -f "$STRING_FILE" || true
    STRING_DECOMPRESSED="${STRING_FILE%.gz}"
    if [ -f "$STRING_DECOMPRESSED" ]; then
        echo "  ✓ STRING database downloaded: $STRING_DECOMPRESSED"
    fi
else
    echo "  ✗ Failed to download STRING database"
fi
echo ""

# 2. Download IntAct database (Human)
echo "2. Downloading IntAct database (Human interactions)..."
INTACT_URL="https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact-micluster.txt.zip"
INTACT_FILE="intact_human_interactions.zip"

if command -v wget &> /dev/null; then
    wget -q --show-progress -O "$INTACT_FILE" "$INTACT_URL"
elif command -v curl &> /dev/null; then
    curl -L --progress-bar -o "$INTACT_FILE" "$INTACT_URL"
fi

# Extract IntAct file
if [ -f "$INTACT_FILE" ]; then
    echo "  Extracting IntAct file..."
    unzip -q -o "$INTACT_FILE" -d "$DATABASE_DIR" 2>/dev/null || {
        # If unzip fails, try downloading the uncompressed version
        echo "  Trying uncompressed IntAct file..."
        rm -f "$INTACT_FILE"
        INTACT_URL="https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact.txt"
        INTACT_FILE="intact_human_interactions.txt"
        if command -v wget &> /dev/null; then
            wget -q --show-progress -O "$INTACT_FILE" "$INTACT_URL"
        elif command -v curl &> /dev/null; then
            curl -L --progress-bar -o "$INTACT_FILE" "$INTACT_URL"
        fi
    }
    if [ -f "intact-micluster.txt" ] || [ -f "$INTACT_FILE" ]; then
        if [ -f "intact-micluster.txt" ]; then
            mv "intact-micluster.txt" "intact_human_interactions.txt"
        fi
        echo "  ✓ IntAct database downloaded: intact_human_interactions.txt"
    fi
    # Clean up zip file if it exists
    rm -f "$INTACT_FILE"
else
    echo "  ✗ Failed to download IntAct database"
fi
echo ""

# 3. Download BioGRID database (Human-specific)
echo "3. Downloading BioGRID database (Human)..."
BIOGRID_URL="https://downloads.thebiogrid.org/BioGRID/Latest-Release/BIOGRID-ORGANISM-Homo_sapiens-LATEST.tab3.zip"
BIOGRID_FILE="biogrid_human_interactions.zip"

if command -v wget &> /dev/null; then
    wget -q --show-progress -O "$BIOGRID_FILE" "$BIOGRID_URL"
elif command -v curl &> /dev/null; then
    curl -L --progress-bar -o "$BIOGRID_FILE" "$BIOGRID_URL"
fi

# Extract BioGRID file
if [ -f "$BIOGRID_FILE" ]; then
    echo "  Extracting BioGRID file..."
    unzip -q -o "$BIOGRID_FILE" -d "$TEMP_DIR" 2>/dev/null
    # Find the extracted file
    EXTRACTED_FILE=$(find "$TEMP_DIR" -name "BIOGRID-ORGANISM-Homo_sapiens-*.tab3.txt" | head -1)
    if [ -f "$EXTRACTED_FILE" ]; then
        mv "$EXTRACTED_FILE" "biogrid_human_interactions.tab3.txt"
        echo "  ✓ BioGRID database downloaded: biogrid_human_interactions.tab3.txt"
    fi
    # Clean up zip file
    rm -f "$BIOGRID_FILE"
else
    echo "  ✗ Failed to download BioGRID database"
fi
echo ""

# Summary
echo "=========================================="
echo "Download Summary"
echo "=========================================="
echo "Files in database directory:"
ls -lh "$DATABASE_DIR"/*.txt* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Download complete!"
