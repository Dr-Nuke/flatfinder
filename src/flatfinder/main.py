import argparse
from pathlib import Path
import sys
from loguru import logger
import random


class ScraperHost:
    def __init__(self, config):
        self.config = config
        self.scraper_dict = {'flatfox':self.flatfox_scraper,
                             'homegate':self.homegate_scraper,
                             'immoscout':self.immoscout_scraper}

        # todo: load database

    def run_main(self):
        for ind_portal, portal in enumerate(self.config.scrapers.keys()):
            self.run_portal(portal)

    def run_portal(self, portal):
        logger.info(f'{portal}')
        portal_config = config.scrapers.get(portal)
        instances = [list(k.keys())[0] for k in portal_config.get('instances')]
        random.shuffle(instances)
        for ind_instance, instance in enumerate(instances):
            df_scraped = self.scraper_dict[portal](instance)
            return df_scraped


    def scrape(self, portal, instance):

        # make scraper
        # get ads
        # return
    def flatfox_scraper(self,instance):
        pass

    def homegate_scraper(self):
        pass

    def immoscout_scraper(self):
        pass

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
