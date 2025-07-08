#!/usr/bin/env python
"""
Count how many unique CCG types appear in the Bobcat lexicon after
replacing every atomic S with N.
"""
from __future__ import annotations
from lambeq import BobcatParser
from lambeq.bobcat.lexicon import Category, Atom

# ------------------------------------------------------------
# 1. Load the grammar (downloads on first call)
# ------------------------------------------------------------
parser = BobcatParser()               # side-effect: loads the grammar

# ------------------------------------------------------------
# 2. Harvest *all* categories in the lexicon
# ------------------------------------------------------------
def all_lexicon_categories(p: BobcatParser) -> set[Category]:
    cats = set()
    # Access the categories through parser.categories (which is a dict)
    # The values are the categories directly
    for cat in p.parser.categories.values():
        cats.add(cat)
    return cats

original = all_lexicon_categories(parser)    # unique types

# ------------------------------------------------------------
# 3. Helper that swaps atomic S → N (feature preserved)
# ------------------------------------------------------------
def contains_s(cat: Category) -> bool:
    """Recursively check if a category contains S anywhere in its structure."""
    if cat.atomic:
        return cat.atom == Atom.S
    else:
        # For complex categories, check both result and argument
        return contains_s(cat.result) or contains_s(cat.argument)

def s_to_n(cat: Category) -> Category:
    if cat.atomic:
        if cat.atom == Atom.S:  # Check if it's an S atom
            # Create new N category with same features
            if cat.feature:
                # Try to create N with the same feature
                try:
                    return Category.parse(f'N[{cat.feature}]')
                except:
                    # Fallback to plain N if feature parsing fails
                    return Category.parse('N')
            else:
                return Category.parse('N')
        return cat
    else:
        # For complex categories, recursively transform result and argument
        result_converted = s_to_n(cat.result)
        argument_converted = s_to_n(cat.argument)
        
        # If neither result nor argument changed, return original
        if (str(result_converted) == str(cat.result) and 
            str(argument_converted) == str(cat.argument)):
            return cat
            
        # Otherwise try to construct the new category
        try:
            # Build the category string and parse it
            cat_str = f'{result_converted}{cat.dir}{argument_converted}'
            return Category.parse(cat_str)
        except:
            # If parsing still fails, we need to signal that this category
            # contains S but couldn't be transformed properly
            # For now, return a placeholder that's clearly different
            try:
                return Category.parse('TRANSFORM_FAILED')
            except:
                return cat

converted = {s_to_n(c) for c in original}

# ------------------------------------------------------------
# 4. Report
# ------------------------------------------------------------
print(f'Original distinct types: {len(original):5d}')
print(f'After S→N, distinct types: {len(converted):5d}')

# ------------------------------------------------------------
# 5. Show detailed statistics
# ------------------------------------------------------------
# Count atomic S categories
atomic_s_categories = [c for c in original if c.atomic and c.atom == Atom.S]
print(f'Found {len(atomic_s_categories)} atomic S categories')

# Method 1: Count using explicit contains_s check
complex_containing_s_explicit = [c for c in original if c.complex and contains_s(c)]
complex_not_containing_s_explicit = [c for c in original if c.complex and not contains_s(c)]

print(f'\nMethod 1 (explicit contains_s check):')
print(f'  Complex categories containing S: {len(complex_containing_s_explicit)}')
print(f'  Complex categories without S: {len(complex_not_containing_s_explicit)}')

# Method 2: Count using transformation-based check
complex_with_s = []
complex_unchanged = []
for cat in original:
    if cat.complex:
        # Check if this category or its subcategories contain S
        transformed = s_to_n(cat)
        if str(transformed) != str(cat):  # Changed during conversion
            complex_with_s.append((cat, transformed))
        else:
            complex_unchanged.append(cat)

print(f'\nMethod 2 (transformation-based check):')
print(f'  Complex categories containing S: {len(complex_with_s)}')
print(f'  Complex categories without S: {len(complex_unchanged)}')

# Check for discrepancies
if len(complex_containing_s_explicit) != len(complex_with_s):
    print(f'\n⚠️  DISCREPANCY DETECTED!')
    print(f'   Explicit method found: {len(complex_containing_s_explicit)}')
    print(f'   Transformation method found: {len(complex_with_s)}')
    
    # Find categories that differ between methods
    explicit_set = set(complex_containing_s_explicit)
    transform_set = {cat for cat, _ in complex_with_s}
    
    only_in_explicit = explicit_set - transform_set
    only_in_transform = transform_set - explicit_set
    
    if only_in_explicit:
        print(f'   Categories found only by explicit method ({len(only_in_explicit)}):')
        for cat in list(only_in_explicit)[:5]:
            print(f'     {cat}')
    
    if only_in_transform:
        print(f'   Categories found only by transformation method ({len(only_in_transform)}):')
        for cat in list(only_in_transform)[:5]:
            print(f'     {cat}')

print(f'\nTotal complex categories: {len(complex_containing_s_explicit) + len(complex_not_containing_s_explicit)}')

print(f'\nSummary (using explicit method):')
print(f'  Total categories: {len(original)}')
print(f'  Atomic S categories: {len(atomic_s_categories)}')
print(f'  Complex categories with S: {len(complex_containing_s_explicit)}')
print(f'  Total categories affected by S→N: {len(atomic_s_categories) + len(complex_containing_s_explicit)}')
print(f'  Reduction in distinct types: {len(original) - len(converted)}')

# Show some examples of each type
print(f'\nExamples of atomic S categories:')
for cat in atomic_s_categories[:5]:  # Show first 5
    print(f'  {cat} -> {s_to_n(cat)}')

print(f'\nExamples of complex categories containing S:')
for cat in complex_containing_s_explicit[:5]:  # Show first 5
    print(f'  {cat} -> {s_to_n(cat)}')
