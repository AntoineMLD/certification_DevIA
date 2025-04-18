import logging

import scrapy

from france_optique.items import FranceOptiqueItem


class GlassSpiderOptovision(scrapy.Spider):
    name = "glass_spider_optovision"
    allowed_domains = ["www.france-optique.com"]
    start_urls = [
        "https://www.france-optique.com/gravures/fournisseur=789",
        "https://www.france-optique.com/gravures/fournisseur=2069",
        "https://www.france-optique.com/gravures/fournisseur=1407",
        "https://www.france-optique.com/gravures/fournisseur=2397",
        "https://www.france-optique.com/gravures/fournisseur=2644",
        "https://www.france-optique.com/gravures/fournisseur=2414",
    ]

    def parse(self, response):
        self.log(f"Analyse de la page: {response.url}", level=logging.INFO)
        supplier_name = response.xpath("//input[@class='readonly']/@value").get()

        lines = response.xpath(
            './/div[@class="tableau_gravures show-on-large hide-on-med-and-down"]/div'
        )

        for line in lines:

            item = FranceOptiqueItem()

            # Ajoute l'URL source à l'item
            item["source_url"] = response.url

            # Extraction du nom du verre
            glass_name = line.xpath(
                './/div[contains(@class, "td col s4 m4")]/p/text()'
            ).get()
            if not glass_name:
                continue
            item["glass_name"] = glass_name

            # Gravure nasale (gestion image ou texte)
            gravure_nasale_img = line.xpath(
                './/div[contains(@class, "s1")][2]/img/@src'
            ).get()
            gravure_nasale_txt = line.xpath(
                './/div[contains(@class, "s1")][2]/p[@class="gravure_txt"]/b//text()'
            ).getall()
            if gravure_nasale_img:
                item["nasal_engraving"] = gravure_nasale_img
            elif gravure_nasale_txt:
                item["nasal_engraving"] = gravure_nasale_txt
            else:
                item["nasal_engraving"] = None

            # Extraction de l'indice et du matériau
            glass_index = (
                line.xpath('.//div[contains(@class, "s1")][4]/p/text()').get()
                or "pas d'indice"
            )

            material = line.xpath('.//div[contains(@class, "s1")][5]/p/text()').get()
            if not glass_index or not material:
                continue
            item["glass_index"] = glass_index
            item["material"] = material

            # Ajout du nom du fournisseur de verre
            item["glass_supplier_name"] = supplier_name.strip()

            yield item
