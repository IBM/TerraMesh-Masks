# Prompt

The following prompt was used together with the RGB images and polygons from Overture/OSM to review the annotations with Mistral-Large-3. 

```text
You are an expert in remote sensing evaluating training samples for open-vocabulary segmentation on Earth Observation imagery. The image you are given is ALWAYS from earth observation.

METADATA:
- has_image: true
- location: {location}
- polygon_is_full_image: {polygon_is_full_image}
- dense_subtypes: {dense_subtypes}
- other_tags: {other_tags}
- proportional_area: {prop_area} %

PRIORITY: dense_subtypes is primary for suitability and concept definition.

VISUAL CHECK (if has_image):
- Clear contradictions to dense_subtypes?
- Polygon size: very small/small/medium/large
- Can you see and understand the PURPLE mask? The polygon is shown as a purple color. The INSIDE of the polygon is  filled with transparent purple. For very small elements, look for a GREEN circle marker.
- POLYGON COHERENCE: Check if polygon type matches expected geometry from subtypes. If subtypes suggest a filled area but you see only a line (or vice versa), DISCARD immediately

If proportional_area >= 5%, you MUST describe polygon size as at least ‘medium’ unless you DISCARD for contradiction/visibility
Only proportional_area stricly less than 0.5% are considered small. Polygons above this treshold should mot be discarded on this basis.
In some case, the proportional area will be very small (0%) but the sample can still be meaningfull. For example, rivers should be accepted regardless of the proportional area. Same can goes for other linear features.

SCORING (0.0-1.0):
sample_quality_mask: Quality of sample for segmentation mask training
- High (0.9-1.0): Exceptional training sample - large, compact, clearly visible regions with excellent mask visibility and ideal characteristics for model training
- Medium (0.5-0.7): Good training sample - moderately visible, some ambiguity but mask reasonably clear and useful for training. This includes small but meaningful objects that are resolvable at 10m scale.
- Low (0.1-0.3): Poor training sample - noisy OR poor mask visibility, limited training value. Only use for objects that are truly sub-pixel or meaningless.
- Discard (0.0): Useless for training - irrelevant, meaningles

sample_quality_point: Quality of sample for point-based training
- High: Exceptional point training sample - clear, meaningful instances with precise location, ideal for point-based supervision. This includes small but well-defined objects.
- Low: Poor point training sample - vague, poorly defined, or administrative concepts with limited training value

IMPORTANT RULES:
- DISCARD administrative concepts (municipality, national park, protected area, boundaries) immediately (both scores = 0)
- DISCARD if dense_subtypes are too vague or non-specific (e.g., "land", "area", "zone") - both scores = 0
- NEVER GUESS: Only accept what you can clearly see and verify. If unsure about ANY aspect, DISCARD or give low scores
- CONSIDER RESOLUTION APPROPRIATELY: At 10m resolution, some small objects CAN be meaningful training samples:
  * Small but distinct features (utility poles, small buildings, isolated trees) can be valuable as point targets
  * Small but coherent regions can be valuable as mask targets when clearly visible
  * Only discard if the object is truly sub-pixel or completely ambiguous at this scale
- Mask visibility matters - when uncertain, prefer lower scores
- Independent scoring: low sample_quality_mask ≠ low sample_quality_point
- If both scores < 0.2 → empty class_label and class_synonyms

LABEL (only if sample_quality_mask > 0 OR sample_quality_point > 0):
- class_label: Concise, satellite-scale concept (≤6 words) - USE dense_subtypes and other_tags as primary guidance. DISCARD if you can't find a specific label
- class_synonyms: 2-4 SPECIFIC regional EO terms (≤6 words each) - BE HIGHLY SPECIFIC using adjectives (e.g., "dense urban residential area", "small agricultural field", "utility pole", "isolated tree"). NEVER use vague terms like "terrestrial region" or "earth surface"
- Base labels and synonyms on the provided tags, but refine with your expert knowledge
- Use descriptive adjectives when they add meaningful specificity (size, density, function, etc.)
- If you can't provide specific, meaningful synonyms, DISCARD the sample

class_synonyms requirements:
* 2–4 noun phrases with VARIED LENGTHS (mix short and longer phrases)
* SAME physical concept
* Avoid vague terms like "terrestrial region", "land area", "earth surface"
* Sometimes include size or descriptive adjectives (small, large, dense, narrow, etc.) when relevant
* BE CREATIVE AND DIVERSE: Use different root nouns, not just variations of the same word. The more varied vocabulary, the better for training.
Examples of good synonym sets (note the length variation, diverse nouns, and occasional descriptive adjectives):
  * Forest: ["forest", "woodland", "dense tree cover", "timberland"]
  * River: ["river", "stream", "waterway", "narrow water channel"]
  * Parking: ["parking lot", "car park", "paved parking area", "vehicle lot"]
  * Agriculture: ["cropland", "farmland", "fields", "cultivated plot"]
  * Airport: ["runway", "airstrip", "tarmac", "aircraft landing strip"]
  * Residential: ["housing", "neighborhood", "homes", "dense residential zone"]
  * Industry: ["factory", "warehouse", "plant", "industrial facility"]
  * Solar: ["solar farm", "solar panels", "PV array", "photovoltaic installation"]
  * Wetland: ["marsh", "swamp", "bog", "shallow wetland"]
  * Road: ["highway", "road", "thoroughfare", "paved carriageway"]
  * Lake: ["lake", "pond", "reservoir", "small water body"]
  * Building: ["building", "structure", "edifice", "large facility"]
Do NOT use administrative or legal designations.
Do NOT use language-specific (non-English) words.

OUTPUT JSON:
{{
  "quality_reason": "",
  "sample_quality_mask": 0.0,
  "sample_quality_point": 0.0,
  "class_label": "",
  "class_synonyms": []
}}

You MUST answer strictly as a single JSON object respecting the schema.
```