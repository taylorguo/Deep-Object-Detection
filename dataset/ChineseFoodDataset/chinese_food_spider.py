import os, requests
from lxml import etree


main_page = "https://www.douguo.com/caipu/"
fenlei_url = "https://www.douguo.com/caipu/fenlei"
test_noodle_page = "https://www.douguo.com/caipu/面条/"


def get_html(url):

    r = requests.get(url)
    r.encoding = "utf-8"

    # print(r.text)

    html = etree.HTML(r.text)

    return html


def get_classes_urls_dict(classes_page_url):
    classes_page_html = get_html(classes_page_url)
    classes = classes_page_html.xpath("//ul/li/a/@title")
    catalog_url = classes_page_html.xpath("//ul[@class='sortlist clearfix']/li/a/@href")
    
    classes_url = {}
    for k,v in zip(classes, catalog_url):
        classes_url.update({k:"".join((main_page, k))})

    return classes_url


def get_class_page_nums(single_class_url):

    class_page = get_html(single_class_url)
    page_urls = class_page.xpath("//div[@class='pages']/a/@href")

    page_url_prefix = ""
    all_page_num = []
    for i in list(set(page_urls)):
        page_num = int(i.split("/")[-1])
        all_page_num.append(page_num)

        if page_url_prefix != i[:i.rfind("/")]:
            page_url_prefix = i[:i.rfind("/")+1]
        # print(page_url_prefix)

    all_page_num.sort()
    # print(all_page_num)

    new_num = []

    if (len(all_page_num)>3):
        d0 = all_page_num[1]-all_page_num[0]
        d1 = all_page_num[2]-all_page_num[1]
        if d0==d1:
            p_num = all_page_num[0]
            i = 0
            while p_num <= all_page_num[-1]:
                new_num.append(page_url_prefix + str(p_num))
                p_num += d0
    else:
        new_num = all_page_num
    
    return new_num


def get_page_img(page_url):
    page_html = get_html(page_url)
    page_imgs = page_html.xpath("//ul[@id='jxlist']/li/a/img/@src")
    return page_imgs


def get_all_urls(cls_url):

    all_url_list = []
    cls_urls =  get_classes_urls_dict(cls_url)
    for i in cls_urls.values():
        for j in get_class_page_nums(i):
            all_url_list.extend(get_page_img(j))
    
    return all_url_list


def download_img_list(url_list):

    if not os.path.exists("images"):
        os.mkdir("images")
    
    i =1 
    for each in url_list:
        print('正在下载第' + str(i) + '张图片，图片地址:' + str(each))
        try:
            pic = requests.get(each, timeout=10)
        except requests.exceptions.ConnectionError:
            print('【错误】当前图片无法下载')
            continue

        file_name = each.split("/")[-1]
        # dir = 'images/' + 'douguo_{:%Y%m%dT%H%M%S}.jpg'.format(datetime.datetime.now())
        dir = 'images/' + file_name

        with open(dir, 'wb') as fp:
            fp.write(pic.content)

        i += 1


def download_img(url_list):
    if not os.path.exists("images"):
        os.mkdir("images")
    
    i =1 
    for each in url_list:
        
        new_each = each.replace("400x266", "yuan")
        print('正在下载第' + str(i) + '张图片，图片地址:' + str(each))

        try:
            pic = requests.get(new_each, timeout=10)
        except requests.exceptions.ConnectionError:
            print('【错误】当前图片无法下载')
            continue

        file_name = new_each.split("/")[-1]
        # dir = 'images/' + 'douguo_{:%Y%m%dT%H%M%S}.jpg'.format(datetime.datetime.now())
        dir = 'images/' + file_name

        with open(dir, 'wb') as fp:
            fp.write(pic.content)

        i += 1


def main(url):
    c_dict = get_classes_urls_dict(url)
    for each_cls in c_dict.values():
        # print(get_class_page_nums(each_cls))
        for each_page in get_class_page_nums(each_cls):
            # print(get_page_img(each_page))
            download_img(get_page_img(each_page))
            # for each_img in get_page_img(each_page):
                # new_each_img = each_img.replace("400x266", "yuan")
                # print(new_each_img)
                # download_img(each_img)


if __name__ == "__main__":
    main(fenlei_url)