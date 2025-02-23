import logging

import scrapy

from france_optique.items import FranceOptiqueItem


class GlassSpiderFullXPath(scrapy.Spider):
    name = "glass_spider_full_xpath"
    allowed_domains = ["www.france-optique.com"]
    start_urls = [
        "https://www.france-optique.com/gravures/fournisseur=70",
        "https://www.france-optique.com/gravures/fournisseur=521",
    ]

    def parse(self, response):
        self.log(f"Analyse de la page: {response.url}", level=logging.INFO)

        # Récupérer le nom du fournisseur
        supplier_name = response.xpath(
            "/html/body/div[2]/div/div[3]/div/div/div/text()"
        ).get()

        # Sélectionner les lignes avec full XPath
        lines = response.xpath(
            '//div[contains(@class, "row") and contains(@class, "tr")]'
        )

        for line in lines:

            item = FranceOptiqueItem()

            # Ajoute l'URL source à l'item
            item["source_url"] = response.url

            # Extraction du nom du verre avec full XPath
            glass_name = line.xpath('.//div[contains(@class, "td")][4]/p/text()').get()
            if not glass_name or not glass_name.strip():
                continue
            item["glass_name"] = glass_name.strip()

            # Extraction de la gravure nasale
            nasal_engraving = line.xpath('.//img[contains(@src, "nasal/")]').get()
            if not nasal_engraving:
                continue
            item["nasal_engraving"] = nasal_engraving.strip()

            # Extraction de l'indice et du matériau avec full XPath
            glass_index = line.xpath('.//div[@class="td col s1 m1"][4]/p/text()').get()
            material = line.xpath(
                "/html/body/div[2]/div/div[4]/div/div/div[1]/div[4]/div[6]/p"
            ).get()
            if not glass_index or not material:
                continue
            item["glass_index"] = glass_index.strip()
            item["material"] = material.strip()

            # Ajout du nom du fournisseur de verre
            item["glass_supplier_name"] = supplier_name.strip()

            yield item
