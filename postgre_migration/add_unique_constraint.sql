-- Ajout de la contrainte UNIQUE sur la colonne nom de la table traitements
ALTER TABLE traitements ADD CONSTRAINT traitements_nom_key UNIQUE (nom); 