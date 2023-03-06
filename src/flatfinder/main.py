import argparse
from pathlib import Path
import sys
from loguru import logger
import random
import numpy as np
import time
from bs4 import BeautifulSoup
import utils.utils as u
import pandas as pd
import datetime
import requests
import re
import math
import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver import DesiredCapabilities
from selenium.webdriver.support.ui import Select
from webdriver_manager.firefox import GeckoDriverManager


# todo: move the portal scraper into own (sub)-classes
class ScraperHost:
    def __init__(self, config):
        self.config = config
        self.scraper_dict = {'flatfox': self.flatfox_scraper,
                             'homegate': self.homegate_scraper,
                             'immoscout': self.immoscout_scraper}

        self.driver = self.make_webdriver()
        self.current_instance = None
        self.current_portal = None
        self.db = u.load_df_safely(self.config.db_path)
        self.old_ids = self.db.id.values
        self.known = 'already known'
        self.temporary = 'Befristet'
        self.scraping_start = datetime.datetime.now()

    def known_ids(self):
        return set(self.db.id.values)

    def run_main(self):
        for ind_portal, portal in enumerate(self.config.scrapers.keys()):
            self.current_portal = portal
            self.run_portal(portal)
        logger.info('Done. Closing driver.')
        self.driver.quit()
        self.send_mail()

    def send_mail(self):
        fromaddr = self.config.email.get('fromaddr')
        toaddr = self.config.email.get('toaddr')
        msg = MIMEMultipart()
        msg['From'] = self.config.email.get('fromaddr')
        msg['To'] = self.config.email.get('toaddr')
        msg['Subject'] = self.config.email.get('subject')


        intro = self.config.email.get('intro')
        sendcols = ['portal', 'zip', 'city', 'link', 'address',
                    'title', 'brutto', 'rooms', 'floor', 'area', 'from']
        result = self.db[~self.db.id.isin(self.known_ids())][sendcols]

        result_formatted = result.sort_values(by='scrape_time', ascending=False).style.format(
            {'link': u.make_clickable})
        body = result_formatted.hide_index().render()
        # second_table = old_results.hide_index().render()

        msg.attach(MIMEText(intro, 'text'))
        msg.attach(MIMEText(body, 'html'))
        # msg.attach(MIMEText(second_table, 'html'))

        server = smtplib.SMTP(self.config.email.get('smtp_host'),
                              self.config.email.get('smtp_port'))
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(self.config.email.get('fromaddr'),
                     self.config.email.get('pwd'))
        text = msg.as_string()
        server.sendmail(fromaddr, toaddr, text)
        logger.info('email sent')

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
        self.driver.get(url)
        u.sleep_randomly()
        load_more_counter = 0
        max_load_more = 2
        found_new_ads = True  # initialize as true to get the loop running once at least
        previous_ads = pd.DataFrame({'id': []})  # iitialize empty df of "previous ads"

        while (load_more_counter <= max_load_more) and found_new_ads:
            all_ads_on_page = self.get_all_flatfox_ads_on_page()
            # now identify those ids that are new in this run
            new_ids = set(all_ads_on_page.id).difference(previous_ads.id)
            df_new_ids = all_ads_on_page[all_ads_on_page.id.isin(new_ids)]

            if not all(df_new_ids.reason == self.known):  # e.g. if there are some new ads
                page_end_button = self.driver.find_element(by=By.CLASS_NAME, value="css-mrdbyv")
                if page_end_button.text == 'Mehr anzeigen':
                    logger.info(f'found next button. going there.')
                    page_end_button.click()
                    u.sleep_randomly()
                    load_more_counter += 1
                elif page_end_button.text == 'Abonnieren':
                    found_new_ads = False
                    logger.info(f'page end button is not providing more ads, not loading more')
            else:
                found_new_ads = False

            previous_ads = all_ads_on_page
            self.log_found_ads(df_new_ids)

        self.log_found_ads(all_ads_on_page, s=' overall')
        # todo: add zip filtering
        # todo: add travel time filtering
        # todo: add noise geo.admin filtering
        df_flatfox_ads = self.get_list_of_raw_ads(all_ads_on_page)

        # import pickle
        # with open('initial.pk', 'wb') as fh:
        #     pickle.dump(df_flatfox_ads, fh)

        df_flatfox_ads = self.process_flatfox_ads(df_flatfox_ads)
        self.add_to_archive(df_flatfox_ads)
        return

    def add_to_archive(self, df):
        "check in newly scraped ads to the archive"
        self.db = pd.concat([self.db, df], ignore_index=True)
        self.db.to_csv(self.config.db_path,
                       index=False)

    def process_flatfox_ads(self, df):
        # run processing jobs

        n_iter = len(df)
        llogger = u.Looplogger(n_iter, f'processing {n_iter} entries')
        for i, irow in df.sample(frac=1).iterrows():
            if (not irow.scrape) or (irow.raw_ad is None):
                continue

            result = {'index': [i]}
            soup = BeautifulSoup(irow.raw_ad, features="html.parser")

            title_field = soup.find('div', attrs={'class': 'widget-listing-title'})
            df.loc[i, 'address'] = u.html_clean_1(title_field.h2.text).split('-')[0].rstrip(' ')
            df.loc[i, 'title'] = u.html_clean_1(title_field.h1.text)
            tables = soup.find_all('table', attrs={'class': 'table table--rows table--fluid table--fixed table--flush'})
            if len(tables) != 2:
                logger.info(f'expected 2 tables, found {len(tables)} for {irow["href"]}, {irow["fname"]}. ignoring')
                continue

            data = []
            rows = tables[0].find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                data.append([ele for ele in cols if ele])  # Get rid of empty values
            for ele in data:
                if 'bruttomiete' in ele[0].lower():
                    df.loc[i, 'brutto'] = u.extract_price_number_flatfox(ele[1])
                elif 'preiseinheit' in ele[0].lower():
                    df.loc[i, 'price_detail'] = ele[1]
                elif 'nettomiete' in ele[0].lower():
                    df.loc[i, 'netto'] = u.extract_price_number_flatfox(ele[1])
                elif 'nebenkosten' in ele[0].lower():
                    df.loc[i, 'utilities'] = u.extract_price_number_flatfox(ele[1])
                else:
                    logger.info(f'unreckognized field {ele[0]} with value {ele[1]} on ad {irow["href"]}')

            data = []
            rows = tables[1].find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                data.append([ele for ele in cols if ele])  # Get rid of empty values

            for ele in data:
                if 'anzahl zimmer' in ele[0].lower():
                    df.loc[i, 'rooms'] = u.flatfox_fix_rooms(ele[1])
                elif 'besonderes' in ele[0].lower():
                    df.loc[i, 'special'] = ele[1]
                elif 'wohnfläche' in ele[0].lower():
                    df.loc[i, 'area'] = u.flatfox_squaremeters(ele[1])
                elif 'ausstattung' in ele[0].lower():
                    df.loc[i, 'particulars'] = ele[1]
                elif 'bezugstermin' in ele[0].lower():
                    df.loc[i, 'from'] = ele[1]

                elif 'referenz' in ele[0].lower():
                    df.loc[i, 'reference'] = ele[1]
                elif 'etage' in ele[0].lower():
                    df.loc[i, 'floor'] = ele[1]
                elif 'nutzfläche' in ele[0].lower():
                    df.loc[i, 'area_usage'] = ele[1]
                elif 'baujahr' in ele[0].lower():
                    df.loc[i, 'constructed'] = ele[1]
                elif 'webseite' in ele[0].lower():
                    df.loc[i, 'website'] = ele[1]
                elif 'dokumente' in ele[0].lower():
                    pass
                elif 'kubatur' in ele[0].lower():
                    pass
                elif 'Renovationsjahr' in ele[0].lower():
                    pass
                else:
                    logger.info(f'unreckognized field {ele[0]} with value {ele[1]} on ad {irow["link"]}')

            if 'brutto' in result: #todo: rent seems to be missing in output
                df.loc[i, 'rent'] = df.loc[i, 'brutto']
            elif 'netto' in result:
                df.loc[i, 'rent'] = df.loc[i, 'netto']
                if 'utilities' in result:
                    df.loc[i, 'rent'] += df.loc[i, 'utilities']
            else:
                logger.info(f'no rent info can be extracted from {irow["link"]}')

            a = soup.find_all('div', attrs={'class': 'fui-stack'})[1]
            df.loc[i, 'text'] = re.sub(r"^Beschreibung", "", u.html_clean_1(a.find_all('div')[-2].text))
        df = df.drop(columns='raw_ad')
        return df

    def log_found_ads(self, df, s=''):
        "some enhanced and standardized logging"
        n_newly_found = len(df)
        n_already_know = (df.reason == self.known).sum()
        n_befristet = (df.reason == self.temporary).sum()
        logger.info(
            f"found{s} {n_newly_found}, of which {n_already_know} are already known and {n_befristet} are temporary")

    def get_list_of_raw_ads(self, df_ads):
        # given a list of ad adds, process those that are relevant, and merge to master db
        # df_ads must have a 'scrape' (boolean) and 'link' column
        df_ads = df_ads.reset_index(drop=True)
        df_ads['raw_ad'] = None
        last_sleeptime = 0
        for i, row in df_ads.iterrows():
            if row['scrape']:
                last_sleeptime = u.wait_minimum(abs(random.gauss(0, 3)) + 5, last_sleeptime)
                df_ads.loc[i, 'raw_ad'], df_ads.loc[i, 'scrape_time'] = self.scrape_individual_adpage(
                    df_ads.loc[i, 'link'],
                    i, len(df_ads))

        return df_ads

    def scrape_individual_adpage(self, url, i, total):
        # the scraper for an individual ad, given its url
        logger.info(f'scraping adpage {i}/{total}: {url}')
        now = datetime.datetime.now().replace(microsecond=0)
        tries_remaining = 3
        while tries_remaining > 0:
            content = requests.get(url)
            tries_remaining -= 1
            if content.status_code == 200:
                result = content.text
                break

        if content.status_code != 200:
            logger.info(f'url returned status code {content.status_code}: {url}')
            result = None

        # todo: add failure case
        return result, now

    def get_all_flatfox_ads_on_page(self):
        # given the page the driver is currently seeing, get all ads as a dataframe

        html_from_page = self.driver.page_source
        soup = BeautifulSoup(html_from_page, 'html.parser')
        adlist = soup.find('div', attrs={'class': 'search-result'})
        if len(adlist) != 1:
            logger.error(f'exactly one adlist should be ther, but found {len(adlist)}')
            # todo: catch-action
        ads = adlist.find_all('div', attrs={'class': 'listing-thumb'})
        if not ads:
            logger.error(f'found no ads on this page')
        else:
            logger.info(f'found {len(ads)} ads')

        dfs = []
        base = 'https://flatfox.ch'
        known_ids = self.known_ids()
        for i, element in enumerate(ads):
            if element.a is None:
                continue  # weired things appearing in ads list. todo: investigate & fix
            key = element.a['href']
            result = {'id': [key]}
            if key in known_ids:
                scrape = False
                archive = False
                reason = 'already known'

            elif self.temporary in element.a.find('div', 'attributes').text:
                scrape = False
                archive = True
                reason = self.temporary
            else:
                scrape = True
                archive = True
                reason = None
                location = element.header.a.h2.find('span').text
                zip_n_city = re.search(r'(\d{4})\s(.*)', location)
                zip = zip_n_city.groups()[0]
                city = zip_n_city.groups()[1]
                result.update({'zip': zip, 'city': city})
            link = base + key
            result.update({
                'scrape': [scrape],
                'archive': [archive],
                'reason': reason,
                'link': link,
                'portal': self.current_portal,
                'instance': self.current_instance})
            dfs.append(pd.DataFrame(result))
        return pd.concat(dfs).reset_index(drop=True)

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
        logger.info(f'using flatfox start url {url}')
        return url

    def homegate_scraper(self):
        pass

    def immoscout_scraper(self):
        pass

    def make_webdriver(self):
        logger.info("making webdriver")
        options = Options()
        options.headless = False
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


def extract_price_number_flatfox(s):
    # extracts int(2455) from "CHF 2’455 pro Monat", as found on flatfox
    regex = r'\d{0,2}’?\d{3}'
    return int(re.sub('’', '', re.search(regex, s).group()))


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
