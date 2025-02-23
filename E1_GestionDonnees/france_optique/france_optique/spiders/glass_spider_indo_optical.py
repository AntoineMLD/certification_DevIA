import logging

import scrapy

from france_optique.items import FranceOptiqueItem


class GlassSpiderIndoOptical(scrapy.Spider):
    name = "glass_spider_indo_optical"
    allowed_domains = ["www.france-optique.com"]
    start_urls = [
        "https://www.france-optique.com/gravures/fournisseur=1958",
        "https://www.france-optique.com/gravures/fournisseur=2217",
        "https://www.france-optique.com/gravures/fournisseur=2532",
        "https://www.france-optique.com/gravures/fournisseur=644",
        "https://www.france-optique.com/gravures/fournisseur=1838",
        "https://www.france-optique.com/gravures/fournisseur=561",
        "https://www.france-optique.com/gravures/fournisseur=711",
        "https://www.france-optique.com/gravures/fournisseur=2395",
        "https://www.france-optique.com/gravures/fournisseur=127",
        "https://www.france-optique.com/gravures/fournisseur=659",
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
            glass_index = line.xpath('.//div[contains(@class, "s1")][4]/p/text()').get()

            material = line.xpath('.//div[contains(@class, "s1")][5]/p/text()').get()
            if not glass_index or not material:
                continue
            item["glass_index"] = glass_index
            item["material"] = material

            # Ajout du nom du fournisseur de verre
            item["glass_supplier_name"] = supplier_name.strip()

            yield item
