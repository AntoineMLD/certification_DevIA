-- Suppression des tables si elles existent déjà
DROP TABLE IF EXISTS verres_traitements;
DROP TABLE IF EXISTS verres;
DROP TABLE IF EXISTS traitements;
DROP TABLE IF EXISTS fournisseurs;
DROP TABLE IF EXISTS materiaux;
DROP TABLE IF EXISTS gammes;
DROP TABLE IF EXISTS series;
DROP TABLE IF EXISTS tags;

-- Création des tables
CREATE TABLE traitements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nom TEXT NOT NULL,
    type TEXT NOT NULL
);

CREATE TABLE fournisseurs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nom TEXT NOT NULL UNIQUE
);

CREATE TABLE materiaux (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nom TEXT NOT NULL UNIQUE
);

CREATE TABLE gammes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nom TEXT NOT NULL UNIQUE
);

CREATE TABLE series (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nom TEXT NOT NULL UNIQUE
);

CREATE TABLE verres (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nom TEXT NOT NULL,
    variante TEXT,
    hauteur_min INTEGER,
    hauteur_max INTEGER,
    indice REAL,
    gravure TEXT,
    url_source TEXT,
    fournisseur_id INTEGER,
    materiau_id INTEGER,
    gamme_id INTEGER,
    serie_id INTEGER,
    FOREIGN KEY (fournisseur_id) REFERENCES fournisseurs(id),
    FOREIGN KEY (materiau_id) REFERENCES materiaux(id),
    FOREIGN KEY (gamme_id) REFERENCES gammes(id),
    FOREIGN KEY (serie_id) REFERENCES series(id)
);

CREATE TABLE verres_traitements (
    verre_id INTEGER,
    traitement_id INTEGER,
    PRIMARY KEY (verre_id, traitement_id),
    FOREIGN KEY (verre_id) REFERENCES verres(id),
    FOREIGN KEY (traitement_id) REFERENCES traitements(id)
);

-- Création de la table tags
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    verre_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (verre_id) REFERENCES verres(id)
);

-- Création d'un index pour optimiser les recherches par tag
CREATE INDEX idx_tags_verre_id ON tags(verre_id);
CREATE INDEX idx_tags_tag ON tags(tag);

-- Insertion des données depuis la table enhanced

-- 1. Insertion des traitements uniques
INSERT INTO traitements (nom, type)
SELECT 'Protection' as nom, 'protection' as type
WHERE EXISTS (SELECT 1 FROM enhanced WHERE traitement_protection = 'YES')
UNION
SELECT 'Photochromique' as nom, 'photochromique' as type
WHERE EXISTS (SELECT 1 FROM enhanced WHERE traitement_photochromique = 'YES');

-- 2. Insertion des fournisseurs uniques
INSERT INTO fournisseurs (nom)
SELECT DISTINCT fournisseur FROM enhanced WHERE fournisseur IS NOT NULL;

-- 3. Insertion des matériaux uniques
INSERT INTO materiaux (nom)
SELECT DISTINCT materiau FROM enhanced WHERE materiau IS NOT NULL;

-- 4. Insertion des gammes uniques
INSERT INTO gammes (nom)
SELECT DISTINCT gamme FROM enhanced WHERE gamme IS NOT NULL;

-- 5. Insertion des séries uniques
INSERT INTO series (nom)
SELECT DISTINCT serie FROM enhanced WHERE serie IS NOT NULL;

-- 6. Insertion des verres
INSERT INTO verres (
    nom,
    variante,
    hauteur_min,
    hauteur_max,
    indice,
    gravure,
    url_source,
    fournisseur_id,
    materiau_id,
    gamme_id,
    serie_id
)
SELECT 
    e.nom_du_verre,
    e.variante,
    e.hauteur_min,
    e.hauteur_max,
    e.indice,
    e.gravure,
    e.url_source,
    f.id as fournisseur_id,
    m.id as materiau_id,
    g.id as gamme_id,
    s.id as serie_id
FROM enhanced e
LEFT JOIN fournisseurs f ON e.fournisseur = f.nom
LEFT JOIN materiaux m ON e.materiau = m.nom
LEFT JOIN gammes g ON e.gamme = g.nom
LEFT JOIN series s ON e.serie = s.nom;

-- 7. Insertion des relations verres-traitements
INSERT INTO verres_traitements (verre_id, traitement_id)
SELECT DISTINCT v.id, t.id
FROM verres v
JOIN enhanced e ON v.nom = e.nom_du_verre
JOIN traitements t
WHERE (t.type = 'protection' AND e.traitement_protection = 'YES')
   OR (t.type = 'photochromique' AND e.traitement_photochromique = 'YES');

-- 8. Insertion des tags
INSERT INTO tags (verre_id, tag)
SELECT v.id, e.tag
FROM verres v
JOIN enhanced e ON v.nom = e.nom_du_verre
WHERE e.tag IS NOT NULL;

-- Création des index pour optimiser les performances
CREATE INDEX idx_verres_nom ON verres(nom);
CREATE INDEX idx_verres_fournisseur ON verres(fournisseur_id);
CREATE INDEX idx_verres_materiau ON verres(materiau_id);
CREATE INDEX idx_verres_gamme ON verres(gamme_id);
CREATE INDEX idx_verres_serie ON verres(serie_id); 