# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class FranceOptiqueItem(scrapy.Item):
    source_url = scrapy.Field()
    glass_name = scrapy.Field()
    nasal_engraving = scrapy.Field()
    glass_index = scrapy.Field()
    material = scrapy.Field()
    glass_supplier_name = scrapy.Field()
    image_engraving = scrapy.Field()
    pass
