import json
import yaml
from config.paths import DATA_DIR


def main():
    with open(DATA_DIR / 'watch_list.json') as f:
        data = json.load(f)

    watch_list = []
    data = data['data']['stocks']

    for item in data:
        if item['marketplace'] == 'US':
            watch_list.append({
                'symbol': item['symbol'],
                'name': item['name'],
                'exchange': item['exchange'],
                'type': item['type'],
            })

    with open(DATA_DIR / 'watch_list.yml', 'w') as f:
        yaml.dump(watch_list, f, encoding='utf-8', allow_unicode=True)

    with open(DATA_DIR / 'tickers.txt', 'w') as f:
        for item in watch_list:
            f.write(f'{item["symbol"]}\n')

    print(f'Watch list saved to {DATA_DIR / "watch_list.yml"}')

if __name__ == '__main__':
    main()