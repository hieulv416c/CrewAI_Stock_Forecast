# import requests
# import pandas as pd
# import yfinance as yf
# from datetime import datetime

# API_KEY = '95466b2292057a10832d9bef0f5d6230'
# url = 'http://api.mediastack.com/v1/news'

# # 1. Lấy danh sách ngày giao dịch từ 2024-12-29 đến hôm nay
# ticker = yf.Ticker("AAPL")
# today = datetime.now().strftime('%Y-%m-%d')
# hist = ticker.history(start="2024-12-29", end=today)
# trading_days = [d.strftime('%Y-%m-%d') for d in hist.index]

# # 2. Đọc file đã crawl, lấy danh sách ngày đã có dữ liệu
# try:
#     df_old = pd.read_csv("apple_news_mediastack_6months.csv")
#     # Đảm bảo cột ngày đúng định dạng, chỉ lấy phần ngày (không lấy giờ nếu có)
#     old_days = set(pd.to_datetime(df_old['date']).dt.strftime('%Y-%m-%d'))
# except Exception as e:
#     print("Không tìm thấy file cũ hoặc file lỗi, sẽ crawl tất cả các ngày.")
#     old_days = set()

# # 3. Chỉ lấy những ngày chưa có dữ liệu
# new_days = [day for day in trading_days if day not in old_days]
# print(f"Sẽ crawl {len(new_days)} ngày mới (chưa có trong file cũ).")

# results = []

# for day in new_days:
#     params = {
#         'access_key': API_KEY,
#         'keywords': 'Apple',
#         'languages': 'en',
#         'date': day,
#         'limit': 30,
#         'offset': 0,
#     }
#     response = requests.get(url, params=params)
#     data = response.json()
#     news = data.get('data', [])
#     print(f"{day}: Lấy được {len(news)} bài báo")

#     for article in news:
#         results.append({
#             'date': article.get('published_at'),
#             'title': article.get('title'),
#             'description': article.get('description'),
#             'source': article.get('source'),
#             'url': article.get('url'),
#             'content': article.get('description')
#         })

# # 4. Nếu có kết quả mới thì nối vào file cũ, tránh trùng lặp
# if results:
#     df_new = pd.DataFrame(results)
#     if old_days:
#         # Gộp với file cũ, không trùng dòng
#         df_full = pd.concat([df_old, df_new], ignore_index=True)
#         df_full.drop_duplicates(subset=["date", "title", "url"], inplace=True)
#         df_full.to_csv("apple_news_mediastack_6months.csv", index=False)
#         print(f"Đã cập nhật file apple_news_mediastack_6months.csv (tổng {len(df_full)} bài báo).")
#     else:
#         # Chưa có file cũ, tạo mới luôn
#         df_new.to_csv("apple_news_mediastack_6months.csv", index=False)
#         print(f"Đã lưu {len(df_new)} bài báo vào file apple_news_mediastack_6months.csv")
# else:
#     print("Không có bài báo mới nào để lưu.")

# import requests
# import pandas as pd

# API_KEY = '7f1d2cb8434128a56469b29588f085cb'
# url = 'http://api.mediastack.com/v1/news'

# # Đọc ngày còn thiếu từ missingdays.csv
# df_missing = pd.read_csv('missingdays.csv')
# missing_days = df_missing['0'].astype(str).tolist()

# results = []
# got_days = []  # Những ngày đã lấy được bài báo

# for day in missing_days:
#     params = {
#         'access_key': API_KEY,
#         'keywords': 'Apple',
#         'languages': 'en',
#         'date': day,
#         'limit': 30,
#         'offset': 0,
#     }
#     try:
#         response = requests.get(url, params=params)
#         data = response.json()
#         news = data.get('data', [])
#         print(f"{day}: Lấy được {len(news)} bài báo")

#         if news:
#             got_days.append(day)  # Đánh dấu ngày đã lấy được bài báo

#         for article in news:
#             results.append({
#                 'date': article.get('published_at'),
#                 'title': article.get('title'),
#                 'description': article.get('description'),
#                 'source': article.get('source'),
#                 'url': article.get('url'),
#                 'content': article.get('description')
#             })
#     except Exception as e:
#         print(f"Lỗi ở ngày {day}: {e}")

# # Lưu dữ liệu bài báo mới vào file paper.csv
# if results:
#     df_new = pd.DataFrame(results)
#     try:
#         df_old = pd.read_csv('paper.csv')
#         df_full = pd.concat([df_old, df_new], ignore_index=True)
#         df_full.drop_duplicates(subset=['date', 'title', 'url'], inplace=True)
#         df_full.to_csv('paper.csv', index=False)
#         print(f"Đã cập nhật file paper.csv (tổng {len(df_full)} bài báo).")
#     except FileNotFoundError:
#         df_new.to_csv('paper.csv', index=False)
#         print(f"Đã lưu {len(df_new)} bài báo vào file paper.csv.")
# else:
#     print("Không có bài báo mới nào để lưu.")

# # Xóa những ngày đã crawl được khỏi missingdays.csv
# if got_days:
#     df_remain = df_missing[~df_missing['0'].isin(got_days)]
#     df_remain.to_csv('missingdays.csv', index=False)
#     print(f"Đã xóa {len(got_days)} ngày đã crawl khỏi missingdays.csv (còn lại {len(df_remain)} ngày).")
# else:
#     print("Không có ngày nào được xoá khỏi missingdays.csv.")



#----------------------------------newsapiimport requests
# import pandas as pd
# import requests as re
# API_KEY = '65ff844dce314c4fb69af981ee1813c4'
# url = 'https://newsapi.org/v2/everything'

# # Đọc ngày còn thiếu từ missingdays.csv
# df_missing = pd.read_csv('missingdays.csv')
# missing_days = df_missing['0'].astype(str).tolist()

# results = []
# got_days = []

# for day in missing_days:
#     params = {
#         'q': 'Apple',
#         'from': day,
#         'to': day,
#         'language': 'en',
#         'sortBy': 'relevancy',
#         'pageSize': 100,
#         'apiKey': API_KEY
#     }
#     try:
#         response = re.get(url, params=params)
#         data = response.json()

#         articles = data.get('articles', [])
#         print(f"{day}: Lấy được {len(articles)} bài báo")

#         if articles:
#             got_days.append(day)

#         for article in articles:
#             results.append({
#                 'date': article.get('publishedAt'),
#                 'title': article.get('title'),
#                 'description': article.get('description'),
#                 'source': article.get('source', {}).get('name'),
#                 'url': article.get('url'),
#                 'content': article.get('content')
#             })
#     except Exception as e:
#         print(f"Lỗi ở ngày {day}: {e}")

# # Lưu dữ liệu bài báo mới vào file paper.csv
# if results:
#     df_new = pd.DataFrame(results)
#     try:
#         df_old = pd.read_csv('paper.csv')
#         df_full = pd.concat([df_old, df_new], ignore_index=True)
#         df_full.drop_duplicates(subset=['date', 'title', 'url'], inplace=True)
#         df_full.to_csv('paper.csv', index=False)
#         print(f"Đã cập nhật file paper.csv (tổng {len(df_full)} bài báo).")
#     except FileNotFoundError:
#         df_new.to_csv('paper.csv', index=False)
#         print(f"Đã lưu {len(df_new)} bài báo vào file paper.csv.")
# else:
#     print("Không có bài báo mới nào để lưu.")

# # Xóa những ngày đã crawl được khỏi missingdays.csv
# if got_days:
#     df_remain = df_missing[~df_missing['0'].isin(got_days)]
#     df_remain.to_csv('missingdays.csv', index=False)
#     print(f"Đã xóa {len(got_days)} ngày đã crawl khỏi missingdays.csv (còn lại {len(df_remain)} ngày).")
# else:
#     print("Không có ngày nào được xoá khỏi missingdays.csv.")


#------------------------------------------------Finhub

import pandas as pd
import requests
from datetime import datetime

API_KEY = 'd10souhr01qse6ldqo20d10souhr01qse6ldqo2g'
url = 'https://finnhub.io/api/v1/company-news'
symbol = 'AAPL'  # hoặc bất cứ mã nào bạn muốn

# Đọc ngày còn thiếu từ missingdays.csv
df_missing = pd.read_csv('missingdays.csv')
missing_days = df_missing['0'].astype(str).tolist()

results = []
got_days = []

for day in missing_days:
    params = {
        'symbol': symbol,
        'from': day,
        'to': day,
        'token': API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            print(f"{day}: Lỗi API - {data['error']}")
            continue

        articles = data if isinstance(data, list) else []
        print(f"{day}: Lấy được {len(articles)} bài báo")

        if articles:
            got_days.append(day)

        for article in articles:
            # Chuyển timestamp thành ngày chuẩn ISO
            pub_date = datetime.utcfromtimestamp(article.get('datetime')).strftime('%Y-%m-%dT%H:%M:%SZ') if article.get('datetime') else None
            results.append({
                'date': pub_date,
                'title': article.get('headline'),
                'description': article.get('summary'),
                'source': article.get('source'),
                'url': article.get('url'),
                'content': article.get('summary')  # Finnhub không có content đầy đủ, dùng summary
            })
    except Exception as e:
        print(f"Lỗi ở ngày {day}: {e}")

# Lưu dữ liệu bài báo mới vào file paper.csv
if results:
    df_new = pd.DataFrame(results)
    try:
        df_old = pd.read_csv('paper.csv')
        df_full = pd.concat([df_old, df_new], ignore_index=True)
        df_full.drop_duplicates(subset=['date', 'title', 'url'], inplace=True)
        df_full.to_csv('paper.csv', index=False)
        print(f"Đã cập nhật file paper.csv (tổng {len(df_full)} bài báo).")
    except FileNotFoundError:
        df_new.to_csv('paper.csv', index=False)
        print(f"Đã lưu {len(df_new)} bài báo vào file paper.csv.")
else:
    print("Không có bài báo mới nào để lưu.")

# Xóa những ngày đã crawl được khỏi missingdays.csv
if got_days:
    df_remain = df_missing[~df_missing['0'].isin(got_days)]
    df_remain.to_csv('missingdays.csv', index=False)
    print(f"Đã xóa {len(got_days)} ngày đã crawl khỏi missingdays.csv (còn lại {len(df_remain)} ngày).")
else:
    print("Không có ngày nào được xoá khỏi missingdays.csv.")
