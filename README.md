# AMLR

`AMLR` - Auto Machine Learning Report

Create a bealtifull Machine Learning Report 

![](https://img.shields.io/badge/pypi-0.1.1-blue) ![](https://img.shields.io/badge/python-3.7|3.8|3.9-lightblue) ![](https://img.shields.io/badge/Licence-MIT-lightgray) ![](https://img.shields.io/badge/status-alpha-orange) ![](https://img.shields.io/badge/pipeline-passed-green) ![](https://img.shields.io/badge/testing-passing-green)


**Main Features:**

- Building a Header Page
- Insert your `logo` picture over the top of page
- Style to Body tag
- Create Tables using dataframe pandas
- Insert images from graph application such seaborn, matplotlib and any other
- Coloring text and styles separately
- Personalised Footer
- Automatically creates texto to Images and Tables
- Embedded videos from your own site or from Youtube
- Create Frames easily
- Ordered and Unordered and description lists in one command line
- Building Forms with one line command
- TABs are the greate way to show many subjects

## Where to use
    
- Nicely to be used on embedded in Flask applications. You create a dynamic page very easy
- You can also create dynamic pages in your Django applications
- If you have other applications working in a batch mode, you can create many dynamics pages in background
- You don't need to know HTML6, CSS or some stuff like that, you just know Python

<BR><BR>
<hr>

## Install

```shell
pip install pywpb
```

<BR>
<hr>
<BR><BR>

## Table of Contents

  1. [Header](#Header)
  2. [Body](#Body)
  3. [Body Methods](#Body-Methods)
      * [h (headings)](#h-(headings))
      * [Text Color](#Text-Color)
      * [Write a Text](#Write-a-Text)
      * [Bold Text](#Bold-Text)
      * [Italic Text](#Italic-Text)
      * [Horizontal Line](#Horizontal-Line)
      * [ Writing Table](#Writing-Table)
      * [Incorporate YouTube Videos](#Incorporate-YouTube-Videos)
      * [Your own Videos](#Your-own-Videos)
      * [Creating Frames in your Web Page](#Creating-Frames-in-your-Web-Page)
      * [Lists](#Lists)
         * [Unordered List](#Unordered-List)
         * [Ordered List](#Ordered-List)
         * [Description List](#Description-List)
      * [Frames](#Frames)
      * [Building Forms](#Building-Forms)
      * [Building TABs](#Building-TABs)
  4. [wpIO (web page input and output)](#wpIO-(web-page-input-and-output))
      * [Print Page](#Print-Page)
      * [Load File](#Load-File)
      * [Preview](#Preview)



## Current Modules

To use any `PYWPB` module you must instantiates all of then using the follow command:

### **Header**

There is only one method to create a page. And you need to run it before anything else.

Create a new Web Page with a minimal configuration. You must run this step before any other command. You can choose you lang, if ommited, the default lan will be english.

`sintax`:
```python
from pywpb import pywpb as wpb

h = wpb.header(charset='utf-8', 
                page_size=[21.0, 29.7, 2.0],
                margin=[0.26, 16],
                background='transparent',
                title='Neublis Page without Title',
                logo=False,
                table_width=10,
                table_cellpadding1=4,
                table_cellpadding2=0,
                page_break_before='always',
                col_width=128,
                valign="top",
                td_width=60,
                td_border_top=1,
                td_border_bottom=1,
                td_border_left=1,
                td_border_right=1,
                td_padding_top=0.1,
                td_padding_bottom=0.1,
                td_padding_left=0.1,
                td_padding_right=0,
                logo_url_image='none',
                name_image='none',
                width_image=76,
                height_image=36,
                border_image=0,
                text_width=60,
                text_border=1,
                text_padding=0.1,
                text_logo='none')
```
The values above are default, but you can change to:

**The charset**
- `ASCII` | `ANSI` | `ISO-8869-1` | `UTF-8`

**Page Size**
- `Right`=21.0 | `Left` 29.7 | `Margin` 2.0

**Margin**
- `margin-bottom` | `line-height`

**Background**
- You can change to any wish color 

**Title**
- Title of you page. If you don't give a text, the text above will be provided

`Prints such a string:`

```html
<!DOCTYPE html PUBLIC "-// W3C // DTD XHTML 1.0 Strict // EN"
   "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html>
<head>
	<meta http-equiv="content-type" content="text/html; charset=utf-8"/>
	<title>pywpb Page without Title</title>
	<meta name="generator" content="pywpb - Python Web Page Builder"/>
	<meta name="created" content="2021/02/17T16:44:08"/>
	<meta name="changed" content="0:0:0"/>
	<style type="text/css">
		@page { size: 21.0cm 29.7cm; margin: 2.0cm }
		p { margin-bottom: 0.26cm; line-height: 16%; background: transparent }
	</style>
</head>

```
After `logo` parameter, you can insert values to create a Header page with you logo and text as you wish. You can personalize your page using all features ahead. Remember: the variable `logo` must be changed to `True`.

### **Body**

`sintax`:
```python
from pywpb import pywpb as wpb

b = pwb.body(margin=0, line=10, 
         link='#080', 
		 vlink='#80',
         lang='eng')
```

The values above are default, but you can change to:

**Margin**
- Margin of body. Normally zero.

**Line**
- Line will be `10`%

**Links**
- The color of the links on the page. The first is the `link never clicked`. The other is the `visited Link`.

`Prints such a string:`
```html
	<body lang="eng" link="#080" vlink="#80" dir="ltr">
		<p style="margin-bottom: 0cm; line-height: 10%"><br/>
		</p>
```
<hr>

### **Body Methods**

The `Body Class` has many methods that you can see below.

<hr>

#### **h (headings)**

The `<h1>` to `<h6>` tags are used to define HTML headings.

`<h1>` defines the most important heading. `<h6>` defines the least important heading.

`sintax`:
```python
from pywpb import pywpb as wpb

b = wpb.body() # keeping default values for body

b.h(text='Headings have no text', 
    size=6, background='white', 
    align='left', 
	color='black', 
	shadow=False)
```

**Text**
- The text that you want to write on the headings

**Size**
- As previously presented, this is the size ranging from one to six

**Background**
- The background `color`. Make your choice!

**Align**
- The align of the headings. Can be:

	`left` | `center` | `right` | `ustify`

**Color**
- The `color` of the text to be write on headings. Make your choice!

**Shadow**
- If you want a elegant text with shadow, just change to `True`

`Prints such a string:`
```html
		<hr style="height:1px; border-width:0; color:gray;background-color:gray">
```

<hr>

#### **Text Color**

If you want a text with different color, use this method to write a new text on the page. This method don't insert line feed.

`Sintax:`
```python
from pywpb import pywpb as wpb

b = wpb.body() # keeping default values for body

b.color_text(text='no text provided',
			 color='black')
```
`Prints such a string:`
```html
<span style="color:black">this is my text</span>
```

#### **Write a Text**

This method writes a text, any one that you provided in text argument.

`Sintax`
```python
from pywpb import pywpb as wpb

b = wpb.body() # keeping default values for body

b.w_text('another text to see line feed',3)
```
There is no limit to text. If you have a big text, the better way is to load text using `wpIO`. See `loading text from file`.

*line_feed* is the number of lines will be jump using tag `<BR>`.

`Prints such a string:`
```html
another text to see line feed<BR><BR><BR>
```
<hr>

`Sintax`
```python
from pywpb import pywpb as wpb

b = wpb.body() # keeping default values for bod

text = b.change_text_color(color='blue', text='testing text one changing color', text_to='one')
```

`Prints such a string:`
```html
testing text <span style="color:blue">one </span> changing color<BR><BR>
```
After you changed the color text, you must write the text using `w_text` method.

<hr>

#### **Bold Text**

Made bold text.

`Sintax`
```python
from pywpb import pywpb as wpb

b = wpb.body() # keeping default values for bod

text = b.bold_text(self, text='no text provided', text_to='text')
```

The word `text` will be bold.

<hr>

#### **Italic Text**

Made Italic text.

`Sintax`
```python
from pywpb import pywpb as wpbb

b = wpb.body() # keeping default values for bod

text = b.italic_text(self, text='no text provided', text_to='text')
```
<hr>

#### **Horizontal Line**

The `<hr>` tag defines a thematic break in an HTML page. 
The `<hr>` element is most often displayed as a horizontal rule that is used to separate content (or define a change) in an HTML page.

`Sintax`
```python
from pywpb import pywpb as wpb

b = wpb.body() # keeping default values for bod

b.hline(height=1, border=0, color='gray', background='gray')
```

#### **Writing Table**

`Sintax`
```python
from pywpb import pywpb as wpb

b = wpb.body() # keeping default values for bod

df = pd.DataFrame({ 'id': [1,2,3], 
					'Elapsed_time': [1,21,31],
					'Total_Value': [3.6, 4.2 , 6.1]})

b.w_table(df,alt_text='My Table without footer',foot=False)
```
**data**
- The `data` is a `Pandas DataFrame`.
- Big dataset with many columns or lines is not a good idea.
- May be use `head()` to is good way.

**border** 
- That's the thickness of the border

**Align**
- The align table

**Collapse**
- Usually there are two lines in table, one to cell, other to silhouette the table it self. Default is `collapse`.

**Color**
- The color of the line the table

**Text to the Table**
- Description to the table.
- All tables will be numbered

**Footer**
- If the last line of the table is a footer, this argument must be `True`.

`Prints such a string:`
Table 1 - My Table without footer


![Table](https://github.com/dbranquinho/pywpb/raw/master/readme_files/table_example.png)

<br>
<hr>
<br>

#### **Incorporate YouTube Videos**

This functionality lets you embed a YouTube video player on your website and control the player using JavaScript.

Using the API's JavaScript functions, you can queue videos for playback; play, pause, or stop those videos; adjust the player volume; or retrieve information about the video being played. You can also add event listeners that will execute in response to certain player events, such as a player state change.

```python
from pywpb import pywpb as wpb

filename = 'test'

h = wpb.header()
b = wpb.body()

b.youtube(video_id='rqz-sutSH0c', url='https://www.youtube.com/iframe_api')

i = wpb.wpIO()
i.write_file(filename,h,b)
i.preview(filename)
```

`Prints such a string HTML6:`

```html
<iframe width="320" height="240" 
            src="https://www.youtube.com/embed/rqz-sutSH0c" 
            frameborder="0" allow="accelerometer; autoplay; 
            clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
        </iframe>

        <script>
        var tag = document.createElement('script');

        tag.src = "https://www.youtube.com/iframe_api";
        var firstScriptTag = document.getElementsByTagName('script')[0];
        firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

        var player;
        function onYouTubeIframeAPIReady() {
            player = new YT.Player('player', {
            height: '240',
            width: '320',
            videoId: 'rqz-sutSH0c',
            events: {
                'onReady': onPlayerReady,
                'onStateChange': onPlayerStateChange
            }
            });
        }

        function onPlayerReady(event) {
            event.target.playVideo();
        }

        var done = false;
        function onPlayerStateChange(event) {
            if (event.data == YT.PlayerState.PLAYING && !done) {
            setTimeout(stopVideo, 60);
            done = true;
            }
        }
        function stopVideo() {
            player.stopVideo();
        }
        </script>
```
<hr>

#### **Creating Frames in your Web Page**

On a web page, framing means that a website can be organized into frames. Each frame displays a different HTML document. Sidebar headers and menus do not move when the content frame is scrolled up and down. For developers, frameworks can also be convenient.

If you want to create any other `pywpb` features into a frame, you create you file as you wish and load this file and write to a frame using the follow code.

```python
from pywpb import pywpb as wpb

filename = 'test'

h = wpb.header()
b = wpb.body()
i = wpb.wpIO()


b.dlist(header="Header text to list", 
        itens=[["desc 1", 'item 1','item 2','item 3'],
               ["desc 2", 'item a','item b']])

i.write_file(filename=filename,body=b)

text = i.load_text('test.html')

b.w_frame(frame_id="myframe1", text_line=text)

i.write_file(filename,h,b)
i.preview(filename)


```

#### **Lists**

Lists allow developers to group a set of related items in lists. We have tree types of lists as follow.

##### **Unordered List**

The list items will be marked with bullets (small black circles) by default.


```python
from pywpb import pywpb as wpb

filename = 'test'

h = wpb.header()
b = wpb.body()

b.ulist(header="Header text to unordered list", itens=["item x", 'item y','item z'])
b.olist(header="Header text to ordered list", itens=["item 1", 'item 2','item 3'])
b.dlist(header="Header text to description list", 
        itens=[["desc 1", 'item 1','item 2','item 3'],
               ["desc 2", 'item a','item b']])
               
i = wpb.wpIO()
i.write_file(filename,h,b)
i.preview(filename)
```
##### **Unordered List**

Change just only the call `ulist` to `olist`

##### **Description List**

Change just only the call `ulist` to `dlist`

<BR>
<hr>
<BR>

#### **Building Forms**

There are many forms that can be created using `pywpb`. Many are created every day. So, we're going to show you some of them. In any case use the code bellow to build your own form. just change the `form_id` using the [List of the forms](#List-of-the-forms).

**Registration Forms**

```python
from pywpb import pywpb as wpb

filename = 'test'

p = wpb.header('en')
b = wpb.body()
i = wpb.wpIO()

b.w_forms(form_id='form_submit', text='Texto to form', url_privacy='https://mysite.com/Privacy Policy', bus_name='The Scientist', submit='https://mysite.com/execute_this_form')


i.write_file(filename, page=p, body=b)
i.preview(filename)

```


##### **List of the forms**

| form_id  |  Description  | Parameters |
| ------------------- | ------------------- | -- |
|  form_submit |  Registration Forms | form_id +  text + url_privacy + bus_name|
|  form_sign_up |  Contact Forms | form_id + text + url_privacy + bus_name |
    

#### **Building TABs**

Tabs are perfect for single page web applications, or for web pages capable of displaying different subjects.

```python
from pywpb import pywpb as wpb

filename = 'test'

p = wpb.header('en')
b = wpb.body()
i = wpb.wpIO()

b.w_tabs(title_tab='My TAB',
         text_tab='This is my tab, click on the tab to show text',
         tab_name=['Tab1','Business','New Tab'],
         text_tab_name=['You can write here anything, other HTML commands or links and texts',
                        'The number of tab_name must be the same text_tab_name',
                        'For example, this tab is the sane of the New Tab in tab_name

i.write_file(filename, page=p, body=b)
i.preview(filename)

```

![My TAB Example](https://github.com/dbranquinho/pywpb/raw/master/readme_files/mytab.png)

<BR>
<hr>
<BR>


### **wpIO (web page input and output)**

This method used to create an environment ways to input and output to your page created.

`sintax`:
```python
from pywpb import pywpb as wpb

i = wpb.wpIO()
```

There are no arguments to pass, but you will use the methods of that class.

<hr>

This module used to input and output all this that you create using pywpb. The Methods are:

#### **Print Page**

If you want to see how your page was built, this method will show you all HTML tags in your file with text output.

`sintax`:
```python
from pywpb import pywpb as wpb

p = wpb.creator('en')
b = wpb.body()
i = wpb.wpIO()

print_page(self, page=p, body=b, cfg_css=False):

```

If you have been calling commands that require CSS tags, then the `css_cfg` argument should be filled with True, but we will see that later.

<hr>

#### **Load File**
If you have a big text file to write on you page, this method is a best way to do this. After loaded, you can use the `text` to change colors of some words on the texto or put some words in bold or italic style.

You must write text on you page using `w_text` method.

`sintax`:
```python
from pywpb import pywpb as wpb

i = wpbnb.wpIO()
text = i.load_text('path/file.txt')
```

<hr>

#### **Write File**

You can write your page in disk. It is usefull and necessary to reuse after creation.
Just give the name without `html`. The write method wil do it for you.


```python
from pywpb import pywpb as wpb

filename = 'path/myfile'

p = wpb.creator('en')
b = wpb.body()
i = wpb.wpIO()

i.write_file(filename, page=p, body=b)

```

The `page` and body as the same instance that you created before.


<hr>

#### **Preview**

You can load the page created in you default browser with a single command line.


```python
from pywpb import pywpb as wpb

filename = 'path/myfile'

p = wpb.creator('en')
b = wpb.body()
i = wpb.wpIO()

# you code building page here

i.write_file(filename, page=p, body=b)
i.preview(filename)

```





`enjoi!`
