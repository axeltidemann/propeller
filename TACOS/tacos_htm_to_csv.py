from __future__ import print_function
import sys

from lxml.html import parse

for input_file in sys.argv[1:]:
    with open('{}.csv'.format(input_file[:input_file.find('.htm')]),'w') as file:
        page = parse(input_file)
        tables = page.xpath('body/div/center/table')
        for table in tables:
            rows = table.findall('tr')
            skipped = 0
            for i,row in enumerate(rows[:-2]): # The last two lines are blank + navigation
                children = row.getchildren()
                if children[0].text_content().strip() == '': # Blank first index. No.
                    skipped += 1
                else:
                    print(','.join([ col.text_content().strip().replace(',', ' ').encode('ascii', 'ignore').decode('ascii') 
                                     for col in children ]), file=file)

            print('{}: {}, rows saved to csv format, {} skipped because of blank indices.'.format(input_file, i-skipped, skipped))
        


