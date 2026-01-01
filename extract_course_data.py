import os
import json
import re
import decimal
import mysql.connector
from datetime import date, datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

# Configuration
OUTPUT_DIR = "output/courses"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def json_serial(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj,  decimal.Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def clean_filename(title: str) -> str:
    """Sanitize title for filename"""
    s = str(title).lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_')

def format_currency(amount) -> str:
    if amount is None:
        return "Free"
    return f"${float(amount):.2f}"

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "mctlms")
    )

def fetch_courses():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # 1. Fetch Request: Get all courses with category and SEO data
    query = """
    SELECT 
        c.courseId, c.title, c.description, c.courseCode, c.slug,
        c.metaKeyword, c.metaDescription, c.seoSchema,
        cat.categoryName, cat.categoryDescription
    FROM course c
    LEFT JOIN coursecategory cat ON c.categoryId = cat.categoryId
    WHERE c.isActive = 1
    """
    cursor.execute(query)
    courses = cursor.fetchall()
    
    # Enrich with related data
    for course in courses:
        cid = course['courseId']
        
        # 2. Content
        cursor.execute("""
            SELECT contentDetails, objectives, prerequisites, courseContent 
            FROM coursecontent WHERE courseId = %s
        """, (cid,))
        course['content'] = cursor.fetchall()
        
        # 3. Cost Plans
        cursor.execute("""
            SELECT planName, planPrice, offerPrice, isActive 
            FROM coursecostplan WHERE courseId = %s AND isActive = 1
        """, (cid,))
        course['costs'] = cursor.fetchall()
        
        # 4. Duration
        cursor.execute("""
            SELECT courseDuration, courseDurationType 
            FROM courseduration WHERE courseId = %s
        """, (cid,))
        course['duration'] = cursor.fetchall()
        
    conn.close()
    return courses

def clean_html(text):
    if not text: return ""
    return re.sub(r'<[^>]+>', '', str(text)).strip()

def extract_company_name(path: str) -> str:
    """Extract company name from image path like /images/course/redhat.png"""
    if not path: return ""
    name = path.split('/')[-1].split('.')[0]
    return name.replace('-', ' ').replace('_', ' ').title()

def generate_markdown(course: Dict[str, Any]) -> str:
    # Header
    md = f"# {course['title']}\n\n"
    
    # Metadata Block
    md += "## Course Details\n"
    if course['categoryName']:
        md += f"- **Category**: {course['categoryName']}\n"
    
    # Duration
    if course['duration']:
        d = course['duration'][0]
        md += f"- **Duration**: {d['courseDuration']} {d['courseDurationType']}\n"
        
    # Cost
    if course['costs']:
        prices = []
        for c in course['costs']:
             name = c['planName'] or "Standard"
             price = c['offerPrice'] if c['offerPrice'] is not None else c['planPrice']
             prices.append(f"{name}: {format_currency(price)}")
        md += f"- **Pricing**: {', '.join(prices)}\n"
    
    md += f"- **Course Code**: {course['courseCode']}\n"
    md += f"- **Slug**: {course['slug']}\n\n"
    
    # SEO Metadata
    if course.get('metaKeyword'):
        md += f"**Keywords**: {course['metaKeyword']}\n"
    
    md += "\n" 
    
    # --- JSON Content Extraction ---
    content_rows = course['content']
    c_json = {}
    
    # Try to parse JSON from the courseContent column
    if content_rows and content_rows[0].get('courseContent'):
        try:
            raw_c = content_rows[0]['courseContent']
            if isinstance(raw_c, str):
                raw_c = json.loads(raw_c)
            if isinstance(raw_c, dict) and 'course' in raw_c:
                c_json = raw_c['course']
        except Exception as e:
            print(f"JSON Parse Error for {course['title']}: {e}")

    course_details = c_json.get('courseDetails', {})

    # 1. Overview / Description
    md += "## Overview\n"
    # Priority: JSON description > metaDescription > DB description
    overview_text = ""
    
    # Check JSON description section
    if 'description' in course_details and isinstance(course_details['description'], dict):
        overview_text = clean_html(course_details['description'].get('description', ''))
    
    # Fallbacks
    if not overview_text:
        overview_text = course.get('metaDescription') or course.get('description') or 'No description available.'
    
    md += f"{overview_text}\n\n"
    
    # Additional Description List (e.g. Top Job Roles)
    if 'description' in course_details and isinstance(course_details['description'], dict):
        desc_list = course_details['description'].get('descriptionList', [])
        for item in desc_list:
             if isinstance(item, dict) and item.get('title'):
                 md += f"### {item['title']}\n"
                 if item.get('titleListItems'):
                     for sub in item['titleListItems']:
                         if sub: md += f"- {sub}\n"
                 md += "\n"

    # 2. Key Features
    if 'Key_Features_Impact' in c_json:
        kf_section = c_json['Key_Features_Impact']
        features = kf_section.get('features', [])
        kf_desc = kf_section.get('description', '')
        
        has_features = False
        kf_md = ""
        
        if kf_desc:
            kf_md += f"{kf_desc}\n\n"
            
        if isinstance(features, list) and features:
            for f in features:
                if isinstance(f, dict) and f.get('keyName'):
                    kf_md += f"- {f.get('keyName')}\n"
                    has_features = True
        
        if has_features or kf_desc:
             md += "## Key Features\n" + kf_md + "\n"

    # 3. Objectives / What You Will Learn
    objs = course_details.get('objectives', {})
    if objs:
        obj_md = ""
        if objs.get('description'):
            obj_md += f"{clean_html(objs['description'])}\n\n"
        if objs.get('objectiveList'):
            for item in objs['objectiveList']:
                if item: obj_md += f"- {clean_html(item)}\n"
        
        if obj_md:
            md += "## What You Will Learn\n" + obj_md + "\n"
    elif content_rows and content_rows[0].get('objectives'): # Fallback
         md += "## What You Will Learn\n" + clean_html(content_rows[0]['objectives']) + "\n\n"

    # 4. Audience
    aud = course_details.get('audience', {})
    if aud:
        aud_md = ""
        if aud.get('description'):
            aud_md += f"{clean_html(aud['description'])}\n\n"
        if aud.get('audienceList'):
            for item in aud['audienceList']:
                if item: aud_md += f"- {clean_html(item)}\n"
        
        if aud_md:
            md += "## Who Is This Course For\n" + aud_md + "\n"

    # 5. Eligibility Criteria
    if 'Eligibility_Criteria' in c_json:
        ec = c_json['Eligibility_Criteria']
        if isinstance(ec, list) and ec:
             md += "## Eligibility Criteria\n"
             for item in ec:
                 if isinstance(item, dict) and item.get('toolName'):
                      desc = f": {item['description']}" if item.get('description') else ""
                      md += f"- **{item['toolName']}**{desc}\n"
             md += "\n"

    # 6. Tools Covered
    if 'Covered_Tools' in c_json:
        ct = c_json['Covered_Tools']
        tools = ct.get('tools', [])
        ct_desc = ct.get('description', '')
        
        ct_md = ""
        if ct_desc:
            ct_md += f"{ct_desc}\n\n"
        
        tool_names = [t.get('toolName') for t in tools if isinstance(t, dict) and t.get('toolName')]
        if tool_names:
            ct_md += "**Tools**: " + ", ".join(tool_names) + "\n\n"
            
        if ct_md:
            md += "## Tools Covered\n" + ct_md

    # 7. Hiring Companies
    if 'Hiring_Companies' in c_json:
        companies = c_json['Hiring_Companies']
        if isinstance(companies, list) and companies:
            # Extract names from paths if they are paths
            company_names = [extract_company_name(c) for c in companies if isinstance(c, str) and c]
            if company_names:
                md += "## Hiring Companies\n"
                md += ", ".join(sorted(set(company_names))) + "\n\n"

    # 8. Remuneration / Salary
    rem = course_details.get('remuneration', {})
    if rem and rem.get('remunerationList'):
        md += "## Career & Salary Potential\n"
        for item in rem['remunerationList']:
            if isinstance(item, dict):
                if item.get('name'):
                    md += f"### {item['name']}\n"
                if item.get('remunerationListItems'):
                    for r in item['remunerationListItems']:
                        md += f"{r}\n\n"

    # 9. Highlights
    hl = course_details.get('highlights', {})
    if hl and hl.get('highlightsList'):
        md += "## Course Highlights\n"
        for h in hl['highlightsList']:
            if isinstance(h, dict) and h.get('title'):
                md += f"### {h['title']}\n"
                if h.get('titleListItems'):
                    for sub in h['titleListItems']:
                        md += f"- {sub}\n"
                md += "\n"

    # 10. Prerequisites
    pre = course_details.get('prerequisites', {})
    pre_md = ""
    if pre.get('description'):
        pre_md += f"{clean_html(pre['description'])}\n\n"
    if pre.get('prerequisiteList'):
        for item in pre['prerequisiteList']:
            if item: pre_md += f"- {clean_html(item)}\n"
            
    if pre_md:
        md += "## Prerequisites\n" + pre_md
    elif content_rows and content_rows[0].get('prerequisites'): # Fallback
        md += "## Prerequisites\n" + clean_html(content_rows[0]['prerequisites']) + "\n\n"

    # 11. What's Included
    inc = course_details.get('courseIncluded', {})
    if inc and inc.get('courseIncludedList'):
        md += "## What's Included\n"
        for item in inc['courseIncludedList']:
            if item: md += f"- {item}\n"
        md += "\n"

    # 12. Curriculum (Modules)
    # Priority: courseDetails.content.modules -> courseDetails.scope.scopeList -> contentDetails
    curriculum_found = False
    
    # Try modules (Best source)
    if 'content' in course_details and 'modules' in course_details['content']:
        modules = course_details['content']['modules']
        if isinstance(modules, list) and modules:
            curr_md = ""
            for i, mod in enumerate(modules, 1):
                name = mod.get('name', f"Module {i}")
                
                # Check for quizzes/exams and skip/label if needed
                # User said: "skip the quiz and exams"
                # But let's check items to be sure. If module name is "Assessment-1" maybe skip?
                # User request: "you can skip the quiz and exams"
                
                # Simple heuristic: If name contains "Assessment" or "Exam", maybe capture it or skip?
                # User seemingly wants knowledge. Quizzes might have Q&A which IS knowledge.
                # User said: "skip the quiz and exams" - I will skip modules named "Assessment..." or "Exams"
                if any(x in name.lower() for x in ['assessment', 'exam']):
                    continue
                    
                curr_md += f"### {name}\n"
                if mod.get('moduleDescription'):
                    curr_md += f"{mod['moduleDescription']}\n"
                
                # Items
                items = mod.get('moduleItems', [])
                if items:
                    for item in items:
                         iname = item.get('moduleItemName', 'Lesson')
                         curr_md += f"- {iname}\n"
                curr_md += "\n"
            
            if curr_md:
                md += "## Curriculum\n" + curr_md
                curriculum_found = True

    # Fallback to scopeList (Course 89 structure)
    if not curriculum_found and 'scope' in course_details and 'scopeList' in course_details['scope']:
        items = course_details['scope']['scopeList']
        if items:
             scope_md = ""
             for item in items:
                 name = item.get('name')
                 if not name: continue # Skip empty
                 scope_md += f"### {name}\n"
                 if item.get('scopeListItems'):
                     for sub in item['scopeListItems']:
                          if sub: scope_md += f"- {sub}\n"
                 scope_md += "\n"
             if scope_md:
                 md += "## Curriculum\n" + scope_md
                 curriculum_found = True

    # Fallback to contentDetails
    if not curriculum_found and content_rows and content_rows[0].get('contentDetails'):
        det = clean_html(content_rows[0]['contentDetails'])
        if det:
            md += "## Curriculum\n" + det + "\n\n"
            
    # 13. Why Join This Program
    if 'Why_join_this_program' in c_json:
        why = c_json['Why_join_this_program']
        if isinstance(why, list) and why:
             md += "## Why Join This Program\n"
             for item in why:
                 if isinstance(item, dict) and item.get('title'):
                      md += f"- **{item['title']}**: {item.get('description', '')}\n"
             md += "\n"

    # 14. FAQs
    if 'FAQs' in c_json and 'QAndA' in c_json['FAQs']:
        qa = c_json['FAQs']['QAndA']
        if isinstance(qa, list):
            valid_qa = []
            for item in qa:
                q = item.get('question')
                a = item.get('answer')
                if q and a and q.strip() and a.strip():
                    valid_qa.append(f"**Q: {clean_html(q)}**\n{clean_html(a)}")
            
            if valid_qa:
                md += "## Frequently Asked Questions\n"
                md += "\n\n".join(valid_qa) + "\n\n"

    return md

def main():
    print("Starting course extraction...")
    courses = fetch_courses()
    print(f"Found {len(courses)} active courses.")
    
    for course in courses:
        try:
            md_content = generate_markdown(course)
            filename = f"course_{course['courseId']}_{clean_filename(course['title'])}.md"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)
                
            print(f"Generated: {filename}")
        except Exception as e:
            print(f"Error generating {course.get('title')}: {e}")

    print(f"\nExtraction complete! Files saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
