-- Cases service database (PostgreSQL 13+), owned by M5 (Eya).
-- Source of truth for the tables. The JSON shapes in contracts/*_record.schema.json,
-- sample.schema.json and validation.schema.json mirror these columns.
-- No patient names or identifiers anywhere: samples are identified by a generated code.

CREATE TABLE samples (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code             TEXT NOT NULL UNIQUE,              -- generated, e.g. S-20261010-001 (PROPOSED format)
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    dilution_factor  NUMERIC NOT NULL CHECK (dilution_factor > 0),
    technician       TEXT NOT NULL,                     -- staff pseudonym or ID, never a patient
    status           TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'corrected', 'validated'))
);

CREATE TABLE images (
    id                TEXT PRIMARY KEY,                 -- = inference service image_id (PROPOSED)
    sample_id         UUID NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    file_path         TEXT,                             -- nullable: the file is stored by the inference service (PROPOSED)
    quality_ok        BOOLEAN NOT NULL,
    rejection_reason  TEXT CHECK (rejection_reason IN ('blurry', 'too_dark', 'overexposed')),
    uploaded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (quality_ok = (rejection_reason IS NULL))
);

CREATE TABLE analysis_results (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id            TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    total_count         INTEGER NOT NULL CHECK (total_count >= 0),
    abnormal_count      INTEGER NOT NULL CHECK (abnormal_count >= 0 AND abnormal_count <= total_count),
    rectangles_counted  INTEGER NOT NULL CHECK (rectangles_counted >= 1),
    concentration       NUMERIC NOT NULL CHECK (concentration >= 0),   -- per uL, = result.concentration_per_uL
    confidence          NUMERIC NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    cells_json          JSONB NOT NULL,                                -- = result.cells
    model_version       TEXT
);

-- One row per sample (PROPOSED): PATCH /correct upserts it with validated_at NULL,
-- POST /validate sets validated_at. corrected_* stay NULL if the AI result was accepted unchanged.
CREATE TABLE validations (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id           UUID NOT NULL UNIQUE REFERENCES samples(id) ON DELETE CASCADE,
    corrected_count     INTEGER CHECK (corrected_count >= 0),
    corrected_abnormal  INTEGER CHECK (corrected_abnormal >= 0),
    validator           TEXT NOT NULL,
    comment             TEXT,
    validated_at        TIMESTAMPTZ,
    CHECK (corrected_abnormal IS NULL OR corrected_count IS NULL OR corrected_abnormal <= corrected_count)
);

CREATE INDEX idx_images_sample ON images(sample_id);
CREATE INDEX idx_results_image ON analysis_results(image_id);
