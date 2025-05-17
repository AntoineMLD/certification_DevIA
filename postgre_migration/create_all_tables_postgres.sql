-- enhanced definition
-- staging definition

CREATE TABLE staging (
    id SERIAL PRIMARY KEY,
    source_url TEXT,
    glass_name TEXT,
    nasal_engraving TEXT,
    glass_index TEXT,
    material TEXT,
    glass_supplier_name TEXT,
    image_engraving TEXT
);

CREATE INDEX idx_staging_id ON staging (id);

CREATE TABLE enhanced (
    id SERIAL PRIMARY KEY,
    nom_du_verre TEXT,              -- Nom du verre
    gamme TEXT,                     -- Gamme du verre
    serie TEXT,                     -- Série du verre
    variante TEXT,                  -- Variante du verre
    hauteur_min INTEGER,            -- Hauteur minimale
    hauteur_max INTEGER,            -- Hauteur maximale
    traitement_protection TEXT,     -- Protection
    traitement_photochromique TEXT, -- Photochromique
    materiau TEXT,                  -- Matériau
    indice REAL,                    -- Indice de réfraction du verre
    fournisseur TEXT,              -- Fournisseur
    gravure TEXT,                   -- Gravure
    url_source TEXT,                -- URL source
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tables de référence d'abord (pour les clés étrangères)

CREATE TABLE fournisseurs (
    id SERIAL PRIMARY KEY,
    nom TEXT NOT NULL UNIQUE
);

CREATE TABLE materiaux (
    id SERIAL PRIMARY KEY,
    nom TEXT NOT NULL UNIQUE
);

CREATE TABLE gammes (
    id SERIAL PRIMARY KEY,
    nom TEXT NOT NULL UNIQUE
);

CREATE TABLE series (
    id SERIAL PRIMARY KEY,
    nom TEXT NOT NULL UNIQUE
);

CREATE TABLE traitements (
    id SERIAL PRIMARY KEY,
    nom TEXT NOT NULL,
    type TEXT NOT NULL
);

-- Table principale verres

CREATE TABLE verres (
    id SERIAL PRIMARY KEY,
    nom TEXT NOT NULL,
    variante TEXT,
    hauteur_min INTEGER,
    hauteur_max INTEGER,
    indice REAL,
    gravure TEXT,
    url_source TEXT,
    fournisseur_id INTEGER REFERENCES fournisseurs(id) ON DELETE SET NULL,
    materiau_id INTEGER REFERENCES materiaux(id) ON DELETE SET NULL,
    gamme_id INTEGER REFERENCES gammes(id) ON DELETE SET NULL,
    serie_id INTEGER REFERENCES series(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_verres_nom ON verres(nom);
CREATE INDEX idx_verres_fournisseur ON verres(fournisseur_id);
CREATE INDEX idx_verres_materiau ON verres(materiau_id);
CREATE INDEX idx_verres_gamme ON verres(gamme_id);
CREATE INDEX idx_verres_serie ON verres(serie_id);

-- Tables de liaison

CREATE TABLE verres_traitements (
    verre_id INTEGER REFERENCES verres(id) ON DELETE CASCADE,
    traitement_id INTEGER REFERENCES traitements(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (verre_id, traitement_id)
);

CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    verre_id INTEGER REFERENCES verres(id) ON DELETE CASCADE,
    tags JSONB,  -- PostgreSQL JSONB pour un meilleur stockage/indexation des tags
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tags_verre ON tags(verre_id);
CREATE INDEX idx_tags_jsonb ON tags USING GIN (tags);

-- Fonction pour la mise à jour automatique des timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Ajout de la colonne updated_at et du trigger pour chaque table principale
ALTER TABLE verres ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE;
CREATE TRIGGER update_verres_modtime
    BEFORE UPDATE ON verres
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

ALTER TABLE enhanced ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE;
CREATE TRIGGER update_enhanced_modtime
    BEFORE UPDATE ON enhanced
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Commentaires sur les tables
COMMENT ON TABLE verres IS 'Table principale des verres optiques';
COMMENT ON TABLE tags IS 'Tags associés aux verres pour la recherche';
COMMENT ON TABLE traitements IS 'Types de traitements disponibles pour les verres';
COMMENT ON TABLE fournisseurs IS 'Liste des fournisseurs de verres';
COMMENT ON TABLE materiaux IS 'Types de matériaux utilisés pour les verres';
COMMENT ON TABLE gammes IS 'Gammes de produits disponibles';
COMMENT ON TABLE series IS 'Séries de verres disponibles'; 