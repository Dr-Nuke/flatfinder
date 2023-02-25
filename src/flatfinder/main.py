import argparse
from pathlib import Path
import sys
from loguru import logger
import random
import numpy as np

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver import DesiredCapabilities
from selenium.webdriver.support.ui import Select
from webdriver_manager.firefox import GeckoDriverManager


class ScraperHost:
    def __init__(self, config):
        self.config = config
        self.scraper_dict = {'flatfox': self.flatfox_scraper,
                             'homegate': self.homegate_scraper,
                             'immoscout': self.immoscout_scraper}

        self.driver = self.make_webdriver()
        self.current_instance = None
        self.current_portal = None

        # todo: load database

    def run_main(self):
        for ind_portal, portal in enumerate(self.config.scrapers.keys()):
            self.current_portal = portal
            self.run_portal(portal)

    def run_portal(self, portal):
        logger.info(f'{portal}')
        portal_config = config.scrapers.get(portal)
        instances = list(portal_config.get('instances').keys())
        random.shuffle(instances)
        for ind_instance, instance in enumerate(instances):
            self.current_instance = instance
            logger.info(f'running {self.scraper_dict[portal].__name__} with instance {instance}')
            df_scraped = self.scraper_dict[portal](portal, instance)

    def scrape(self, portal, instance):

        # make scraper
        # get ads
        # return
        pass

    def flatfox_scraper(self, portal, instance):
        url = self.make_random_flatfox_url()
        return df_flatfox

    def make_random_flatfox_url(self):
        conf_portal = self.config.scrapers.get(self.current_portal)
        conf_instance = conf_portal.get('instances').get(self.current_instance)

        url_dict = {k: v for k, v in conf_portal.get('url_base_components').items()}
        url_dict['object_category'] = '&object_category=&'.join([x for x in url_dict['object_category']])  # evil hack

        url_dict.update({k: v for k, v in self.config.search_base_attributes.items()})
        url_dict.update({k: v for k, v in conf_instance.get('url_components').items()})

        geo_variation = self.config.url_randomization.get('geo_variation')
        geo_precision = self.config.url_randomization.get('geo_precision')
        url_dict['west'], url_dict['east'] = randomize_map(url_dict['west'],
                                                           url_dict['east'],
                                                           variation=geo_variation,
                                                           precision=geo_precision)
        url_dict['north'], url_dict['south'] = randomize_map(url_dict['north'],
                                                             url_dict['south'],
                                                             variation=geo_variation,
                                                             precision=geo_precision)

        url_dict['min_price'] = randomize_price(url_dict['min_price'], conf_portal['min_price_variation'])
        url_dict['max_price'] = randomize_price(url_dict['max_price'], conf_portal['max_price_variation'])
        url_dict.pop('max_rooms', None)

        url = conf_portal.get('base_url') + '&'.join([key + '=' + str(val) for key, val in url_dict.items()])
        logger.info(f'usinf flatfox start url {url}')
        return url

    def homegate_scraper(self):
        pass

    def immoscout_scraper(self):
        pass

    def make_webdriver(self):
        logger.info("making webdriver")
        options = Options()
        options.headless = True
        options.add_argument('--window-size=1920,1080')
        options.binary_location = self.config.firefox_executable  # add firefox executable here

        profile = webdriver.FirefoxProfile(str(self.config.firefox_profile))
        # options.set_preference('profile', str(self.config.firefox_profile))
        profile.set_preference("dom.webdriver.enabled", False)
        profile.set_preference('useAutomationExtension', False)
        profile.update_preferences()

        desired = DesiredCapabilities.FIREFOX
        driver = webdriver.Firefox(executable_path=GeckoDriverManager().install(),
                                   options=options,
                                   desired_capabilities=desired,
                                   firefox_profile=profile)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver


def randomize_price(price, variations):
    'mdifies the prices abit'
    return price + random.choice(variations)


def randomize_map(east, west, variation=0.01, precision=6):
    # modifies the geo-tags by a little bit
    variation = abs((east - west)) / east * variation
    new_east = np.round(east * random.uniform(1 + variation, 1 - variation), precision)
    new_west = np.round(west * random.uniform(1 + variation, 1 - variation), precision)
    return new_east, new_west


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config_path",
                        help="path to the global config file. must end with /config/config.py",
                        type=str,
                        required=True)

    parser.add_argument("-w",
                        "--working_directory",
                        help="working director of the execution of this program",
                        default=None
                        )

    args = parser.parse_args()

    lofgilepath = Path(r'C:\coding\flatfinder\logs')
    logger.add(lofgilepath / "logfile{time}.log",
               colorize=True,
               format="<green>{time}</green> <level>{message}</level>")

    configpath = Path(args.config_path)
    logger.info(f'starting main scraping with config file {configpath}')
    if (configpath.parent.name != 'configs') or (configpath.name != 'config.py'):
        logger.error(f"config does not end with /config/config.py but instead is {configpath}. Aborting")

    sys.path.append(str(configpath.parent))
    from configs import config

    scraper_host = ScraperHost(config)
    scraper_host.run_main()
    print('end')
