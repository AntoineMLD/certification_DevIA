import logging

import scrapy

from france_optique.items import FranceOptiqueItem


class GlassSpider(scrapy.Spider):
    name = "glass_spider"
    allowed_domains = ["www.france-optique.com"]
    start_urls = [
        "https://www.france-optique.com/fournisseur/1344-bbgr-optique/gravures",
        "https://www.france-optique.com/fournisseur/2399-adn-optis",
    ]

    def parse(self, response):
        self.log(f"Analyse de la page: {response.url}", level=logging.INFO)
        supplier_name = response.xpath(
            "/html/body/div[2]/div/div[3]/div[2]/h2/text()"
        ).get()

        lines = response.xpath('//*[@id="gravures"]/div[2]//div')

        for line in lines:
            item = FranceOptiqueItem()

            # Ajoute l'URL source à l'item
            item["source_url"] = response.url

            # Extraction du nom du verre
            glass_name = line.css(
                "div.row.tr:not(.group) div.td.col.s3.m3 p::text"
            ).get("")
            if not glass_name.strip():
                continue
            item["glass_name"] = glass_name.strip()

            # Extraction de la gravure nasale
            nasal_engraving = line.xpath(
                './/div[contains(@class, "td")][2]//p[@class="gravure_txt"]/b/text()'
            ).get()
            nasal_engraving = (
                nasal_engraving
                or line.xpath(
                    './/div[contains(@class, "td")][2]/img[contains(@src, "nasal")]/@src'
                ).get()
            )
            if not nasal_engraving:
                continue
            item["nasal_engraving"] = nasal_engraving

            # Extraction de l'indice et du matériau
            glass_index = line.css(
                "div.row.tr:not(.group) div.td.col.s1.m1 p::text"
            ).get()
            material = line.css("div.td.col.s2.m2 p::text").get()
            if not glass_index or not material:
                continue
            item["glass_index"] = glass_index
            item["material"] = material

            # Ajout du nom du fournisseur de verre
            item["glass_supplier_name"] = supplier_name.strip()

            yield item
