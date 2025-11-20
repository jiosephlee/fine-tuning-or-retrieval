import os
import csv
from collections import defaultdict

def main():
    """
    Load all inference probes and display unique values in the inference_type column.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    probes_base_dir = os.path.join(project_root, 'data/probes/inference')
    
    domains = [d for d in os.listdir(probes_base_dir) if os.path.isdir(os.path.join(probes_base_dir, d))]
    
    all_unique_types = set()
    types_by_domain = defaultdict(set)
    type_counts = defaultdict(lambda: defaultdict(int))
    
    for domain in sorted(domains):
        probes_file = os.path.join(probes_base_dir, domain, 'probes_v7.csv')
        if not os.path.exists(probes_file):
            print(f"Probes file not found for {domain}, skipping.")
            continue
        
        with open(probes_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")
        print(f"Total probes: {len(rows)}")
        
        if rows and 'inference_type' in rows[0]:
            unique_types = set()
            for row in rows:
                itype = row['inference_type']
                unique_types.add(itype)
                all_unique_types.add(itype)
                types_by_domain[domain].add(itype)
                type_counts[domain][itype] += 1
            
            print(f"\nUnique inference_type values ({len(unique_types)}):")
            for itype in sorted(unique_types):
                count = type_counts[domain][itype]
                print(f"  - {itype}: {count} probes")
        else:
            print("WARNING: 'inference_type' column not found!")
        
        if rows:
            print(f"\nColumns in dataframe: {list(rows[0].keys())}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY ACROSS ALL DOMAINS")
    print(f"{'='*60}")
    print(f"\nTotal unique inference_type values: {len(all_unique_types)}")
    print("\nAll unique inference_type values:")
    for itype in sorted(all_unique_types):
        domains_with_type = [d for d in sorted(domains) if itype in types_by_domain[d]]
        print(f"  - '{itype}' (in {len(domains_with_type)} domains: {', '.join(domains_with_type)})")

if __name__ == '__main__':
    main()

