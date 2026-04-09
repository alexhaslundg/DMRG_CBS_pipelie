import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings


class ComputeStatsParser:
    """Parse GNU time output from compute_stats files."""
    
    @staticmethod
    def parse_time_output(file_path: Path) -> Dict:
        """
        Parse GNU time output file and extract key metrics.
        
        Returns:
            dict: Parsed compute statistics
        """
        stats = {
            'wall_time_seconds': None,
            'user_time_seconds': None,
            'system_time_seconds': None,
            'cpu_percent': None,
            'max_memory_kb': None,
            'max_memory_gb': None,
            'page_faults_major': None,
            'page_faults_minor': None,
            'exit_status': None,
            'command': None
        }
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse command
            cmd_match = re.search(r'Command being timed: "(.*?)"', content)
            if cmd_match:
                stats['command'] = cmd_match.group(1)
            
            # Parse user time
            user_match = re.search(r'User time \(seconds\): ([\d.]+)', content)
            if user_match:
                stats['user_time_seconds'] = float(user_match.group(1))
            
            # Parse system time
            sys_match = re.search(r'System time \(seconds\): ([\d.]+)', content)
            if sys_match:
                stats['system_time_seconds'] = float(sys_match.group(1))
            
            # Parse CPU percent
            cpu_match = re.search(r'Percent of CPU this job got: (\d+)%', content)
            if cpu_match:
                stats['cpu_percent'] = int(cpu_match.group(1))
            
            # Parse wall clock time (convert to seconds)
            wall_match = re.search(r'Elapsed \(wall clock\) time.*?: ([\d:\.]+)', content)
            if wall_match:
                time_str = wall_match.group(1)
                stats['wall_time_seconds'] = ComputeStatsParser._parse_time_to_seconds(time_str)
            
            # Parse max memory
            mem_match = re.search(r'Maximum resident set size \(kbytes\): (\d+)', content)
            if mem_match:
                kb = int(mem_match.group(1))
                stats['max_memory_kb'] = kb
                stats['max_memory_gb'] = kb / (1024 * 1024)
            
            # Parse page faults
            major_pf = re.search(r'Major \(requiring I/O\) page faults: (\d+)', content)
            if major_pf:
                stats['page_faults_major'] = int(major_pf.group(1))
            
            minor_pf = re.search(r'Minor \(reclaiming a frame\) page faults: (\d+)', content)
            if minor_pf:
                stats['page_faults_minor'] = int(minor_pf.group(1))
            
            # Parse exit status
            exit_match = re.search(r'Exit status: (\d+)', content)
            if exit_match:
                stats['exit_status'] = int(exit_match.group(1))
        
        except Exception as e:
            warnings.warn(f"Error parsing compute stats from {file_path}: {e}")
        
        return stats
    
    @staticmethod
    def _parse_time_to_seconds(time_str: str) -> float:
        """Convert time string (h:mm:ss or m:ss) to seconds."""
        parts = time_str.split(':')
        if len(parts) == 2:  # m:ss
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:  # h:mm:ss
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        else:
            return float(time_str)


class DatabaseCompiler:
    """
    Compile individual calculation runs into a unified database.
    
    Directory structure expected:
    base_dir/
        data/
            H2_dA0.8_HF_cc-pVDZ.json
            H4_dA0.8_DMRG_cc-pVDZ.json
            ...
        compute_stats/
            H2_dA0.8_HF_cc-pVDZ_stats.json
            H4_dA0.8_DMRG_cc-pVDZ_stats.json
            ...
    """
    
    def __init__(self, base_dir: str, system_name: str = "compiled_system"):
        """
        Initialize compiler.
        
        Args:
            base_dir: Base directory containing data/ and compute_stats/ subdirectories
            system_name: Name for the compiled system
        """
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'data'
        self.stats_dir = self.base_dir / 'compute_stats'
        self.system_name = system_name
        self.runs = []
        
        # Verify directory structure
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        if not self.stats_dir.exists():
            warnings.warn(f"Compute stats directory not found: {self.stats_dir}")
        
    def scan_directory(self, pattern: str = "*.json") -> List[Path]:
        """
        Scan data directory for result files matching pattern.
        
        Args:
            pattern: Glob pattern for JSON filenames (default: "*.json")
            
        Returns:
            List of result file paths
        """
        result_files = list(self.data_dir.glob(pattern))
        print(f"Found {len(result_files)} result files in {self.data_dir}")
        return result_files
    
    def parse_run_name(self, filename: str) -> Dict:
        """
        Parse result filename to extract metadata.
        
        Expected formats:
        - H2_dA0.8_HF_cc-pVDZ.json
        - H4_dA0.8_DMRG_cc-pVDZ.json
        - N2_R1.09_FCI_aug-cc-pVTZ.json
        - H_chain_dA0.8_n4_DMRG_cc-pVQZ.json
        
        Returns:
            dict: Parsed metadata (geometry_key, method, basis, base_name)
        """
        # Remove .json extension
        base_name = filename.replace('.json', '')
        parts = base_name.split('_')
        
        # Try to identify method (HF, FCI, DMRG, CCSDT, MP2)
        methods = ['HF', 'FCI', 'DMRG', 'CCSDT', 'MP2']
        method = None
        method_idx = None
        
        for i, part in enumerate(parts):
            if part in methods:
                method = part
                method_idx = i
                break
        
        if method is None:
            raise ValueError(f"Could not identify method in filename: {filename}")
        
        # Basis set is everything after method
        basis = '_'.join(parts[method_idx + 1:])
        
        # Geometry is everything before method
        geometry_parts = parts[:method_idx]
        
        # Parse geometry to create key
        geometry_key = self._create_geometry_key(geometry_parts)
        
        return {
            'geometry_key': geometry_key,
            'method': method,
            'basis': basis,
            'base_name': base_name,
            'raw_filename': filename
        }
    
    def _create_geometry_key(self, parts: List[str]) -> str:
        """
        Create standardized geometry key from parsed parts.
        
        Examples:
        - ['H2', 'dA0.8'] -> 'dA_0.80_n2' (assuming default n=2)
        - ['H4', 'dA0.8'] -> 'dA_0.80_n4'
        - ['N2', 'R1.09'] -> 'R_1.09'
        - ['H', 'chain', 'dA0.8', 'n4'] -> 'dA_0.80_n4'
        """
        # Look for dA, dB, n, R patterns
        dA = None
        dB = None
        n = None
        R = None
        
        # Check for HX or NX pattern (e.g., H2, H4, H6, N2)
        for part in parts:
            if part.startswith('H') and len(part) > 1 and part[1:].isdigit():
                n = int(part[1:])
            elif part.startswith('N') and len(part) > 1 and part[1:].isdigit():
                n = int(part[1:])
            elif part.startswith('dA'):
                dA = float(part[2:])
            elif part.startswith('dB'):
                dB = float(part[2:])
            elif part.startswith('n') and part[1:].isdigit():
                n = int(part[1:])
            elif part.startswith('R'):
                R = float(part[1:])
        
        # Build geometry key
        if R is not None:
            # Diatomic format
            return f"R_{R:.2f}"
        elif dA is not None:
            # Chain format
            if n is None:
                n = 2  # Default for H2
            if dB is not None:
                return f"dA_{dA:.2f}_dB_{dB:.2f}_n{n}"
            else:
                return f"dA_{dA:.2f}_n{n}"
        else:
            # Fallback: join all parts
            return '_'.join(parts)
    
    def compile_database(self, pattern: str = "*.json", 
                        output_file: Optional[str] = None) -> Dict:
        """
        Compile all runs into unified database.
        
        Args:
            pattern: Glob pattern for result files (default: "*.json")
            output_file: Optional path to save JSON output
            
        Returns:
            dict: Compiled database in CBS analyzer format
        """
        result_files = self.scan_directory(pattern)
        
        # Initialize database structure
        database = {
            "system_info": {
                "system_name": self.system_name,
                "total_runs": 0,
                "compilation_timestamp": None
            }
        }
        
        # Track all unique values for system_info
        all_basis_sets = set()
        all_methods = set()
        all_geometries = set()
        
        successful_runs = 0
        failed_runs = 0
        
        for result_file in result_files:
            try:
                # Parse filename
                run_info = self.parse_run_name(result_file.name)
                geometry_key = run_info['geometry_key']
                method = run_info['method']
                basis = run_info['basis']
                base_name = run_info['base_name']
                
                # Load results JSON
                with open(result_file, 'r') as f:
                    results = json.load(f)
                
                # Load compute stats (try both .json and .txt extensions, with _stats suffix)
                compute_stats = {}
                stats_file_json = self.stats_dir / f"{base_name}_stats.json"
                stats_file_txt = self.stats_dir / f"{base_name}_stats.txt"
                
                if stats_file_json.exists():
                    # Assume JSON format contains parsed stats
                    with open(stats_file_json, 'r') as f:
                        compute_stats = json.load(f)
                elif stats_file_txt.exists():
                    # Parse GNU time output
                    compute_stats = ComputeStatsParser.parse_time_output(stats_file_txt)
                else:
                    warnings.warn(f"No compute stats found for {base_name}")
                
                # Initialize geometry entry if needed
                if geometry_key not in database:
                    database[geometry_key] = {}
                
                # Initialize basis entry if needed
                if basis not in database[geometry_key]:
                    database[geometry_key][basis] = {}
                
                # Extract method data from results
                method_data = self._extract_method_data(results, geometry_key, method, basis)
                
                # Merge compute stats
                if compute_stats:
                    # Handle both parsed and raw compute stats formats
                    wall_time = compute_stats.get('wall_time_seconds') or compute_stats.get('wall_time')
                    peak_mem = compute_stats.get('max_memory_gb') or compute_stats.get('peak_memory_gb')
                    
                    if wall_time:
                        method_data['calculation_time'] = wall_time
                    if peak_mem:
                        method_data['peak_memory_gb'] = peak_mem
                    
                    method_data['compute_stats'] = {
                        'user_time': compute_stats.get('user_time_seconds') or compute_stats.get('user_time'),
                        'system_time': compute_stats.get('system_time_seconds') or compute_stats.get('system_time'),
                        'cpu_percent': compute_stats.get('cpu_percent'),
                        'page_faults_major': compute_stats.get('page_faults_major'),
                        'exit_status': compute_stats.get('exit_status')
                    }
                
                # Add to database
                database[geometry_key][basis][method] = method_data
                # Track metadata
                all_basis_sets.add(basis)
                all_methods.add(method)
                all_geometries.add(geometry_key)
                successful_runs += 1
                
            except Exception as e:
                warnings.warn(f"Error processing {result_file.name}: {e}")
                failed_runs += 1
                continue
        
        # Update system_info
        database["system_info"].update({
            "total_runs": successful_runs,
            "failed_runs": failed_runs,
            "basis_sets": sorted(list(all_basis_sets)),
            "methods": sorted(list(all_methods)),
            "geometries": sorted(list(all_geometries))
        })
        
        print(f"\n{'='*60}")
        print(f"Compilation Summary")
        print(f"{'='*60}")
        print(f"Successful runs: {successful_runs}")
        print(f"Failed runs: {failed_runs}")
        print(f"Unique geometries: {len(all_geometries)}")
        print(f"Unique basis sets: {len(all_basis_sets)}")
        print(f"Unique methods: {len(all_methods)}")
        print(f"{'='*60}\n")
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(database, f, indent=2)
            print(f"Database saved to: {output_path}")
        
        return database
    
    def _extract_method_data(self, results: Dict, geometry_key: str, 
                            method: str, basis: str) -> Dict:
        """
        Extract method data from results JSON.
        
        Handles different JSON structures from different run types.
        Specifically extracts von_neumann_entropy_mo for DMRG calculations.
        Also extracts the exact bond length from d_A_values in system_info.
        """
        # Try to find method data in results
        # Structure could be: results[geometry_key][basis][method]
        # or results could be the method data directly
        
        method_data = None
        
        # Try hierarchical structure
        if geometry_key in results:
            if basis in results[geometry_key]:
                if method in results[geometry_key][basis]:
                    method_data = results[geometry_key][basis][method]
        
        # Try flat structure (results is already method data)
        if method_data is None and 'method' in results and results.get('method') == method:
            method_data = results
        
        # Try looking for method in any geometry key
        if method_data is None:
            for geo_key in results.keys():
                if geo_key == 'system_info':
                    continue
                if isinstance(results[geo_key], dict) and basis in results[geo_key]:
                    if method in results[geo_key][basis]:
                        method_data = results[geo_key][basis][method]
                        break
        
        if method_data is None:
            raise ValueError(f"Could not find method data for {method}/{basis} in results")
        
        # Make a copy to avoid modifying original
        extracted_data = method_data.copy()
        
        # Special handling for DMRG: extract von Neumann entropy
        if method == 'DMRG' and 'von_neumann_entropy_mo' in method_data:
            entropy_data = method_data['von_neumann_entropy_mo']
            if isinstance(entropy_data, dict) and 'entropy' in entropy_data:
                # Ensure entropy is at the top level for easy access
                extracted_data['von_neumann_entropy'] = entropy_data['entropy']
                # Keep the full entropy data structure
                extracted_data['von_neumann_entropy_mo'] = entropy_data

        # Extract exact bond length from d_A_values in system_info
        if 'system_info' in results:
            d_A_values = results['system_info'].get('d_A_values', None)
            if d_A_values is not None:
                # d_A_values is a list, take the first value (or could average if multiple)
                if isinstance(d_A_values, list) and len(d_A_values) > 0:
                    extracted_data['bond_length'] = d_A_values[0]
                elif isinstance(d_A_values, (int, float)):
                    extracted_data['bond_length'] = d_A_values
                else:
                    extracted_data['bond_length'] = None
            else:
                extracted_data['bond_length'] = None
        else:
            extracted_data['bond_length'] = None
        
        return extracted_data
# import json
# import re
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple
# import warnings


# class ComputeStatsParser:
#     """Parse GNU time output from compute_stats files."""
    
#     @staticmethod
#     def parse_time_output(file_path: Path) -> Dict:
#         """
#         Parse GNU time output file and extract key metrics.
        
#         Returns:
#             dict: Parsed compute statistics
#         """
#         stats = {
#             'wall_time_seconds': None,
#             'user_time_seconds': None,
#             'system_time_seconds': None,
#             'cpu_percent': None,
#             'max_memory_kb': None,
#             'max_memory_gb': None,
#             'page_faults_major': None,
#             'page_faults_minor': None,
#             'exit_status': None,
#             'command': None
#         }
        
#         try:
#             with open(file_path, 'r') as f:
#                 content = f.read()
            
#             # Parse command
#             cmd_match = re.search(r'Command being timed: "(.*?)"', content)
#             if cmd_match:
#                 stats['command'] = cmd_match.group(1)
            
#             # Parse user time
#             user_match = re.search(r'User time \(seconds\): ([\d.]+)', content)
#             if user_match:
#                 stats['user_time_seconds'] = float(user_match.group(1))
            
#             # Parse system time
#             sys_match = re.search(r'System time \(seconds\): ([\d.]+)', content)
#             if sys_match:
#                 stats['system_time_seconds'] = float(sys_match.group(1))
            
#             # Parse CPU percent
#             cpu_match = re.search(r'Percent of CPU this job got: (\d+)%', content)
#             if cpu_match:
#                 stats['cpu_percent'] = int(cpu_match.group(1))
            
#             # Parse wall clock time (convert to seconds)
#             wall_match = re.search(r'Elapsed \(wall clock\) time.*?: ([\d:\.]+)', content)
#             if wall_match:
#                 time_str = wall_match.group(1)
#                 stats['wall_time_seconds'] = ComputeStatsParser._parse_time_to_seconds(time_str)
            
#             # Parse max memory
#             mem_match = re.search(r'Maximum resident set size \(kbytes\): (\d+)', content)
#             if mem_match:
#                 kb = int(mem_match.group(1))
#                 stats['max_memory_kb'] = kb
#                 stats['max_memory_gb'] = kb / (1024 * 1024)
            
#             # Parse page faults
#             major_pf = re.search(r'Major \(requiring I/O\) page faults: (\d+)', content)
#             if major_pf:
#                 stats['page_faults_major'] = int(major_pf.group(1))
            
#             minor_pf = re.search(r'Minor \(reclaiming a frame\) page faults: (\d+)', content)
#             if minor_pf:
#                 stats['page_faults_minor'] = int(minor_pf.group(1))
            
#             # Parse exit status
#             exit_match = re.search(r'Exit status: (\d+)', content)
#             if exit_match:
#                 stats['exit_status'] = int(exit_match.group(1))
        
#         except Exception as e:
#             warnings.warn(f"Error parsing compute stats from {file_path}: {e}")
        
#         return stats
    
#     @staticmethod
#     def _parse_time_to_seconds(time_str: str) -> float:
#         """Convert time string (h:mm:ss or m:ss) to seconds."""
#         parts = time_str.split(':')
#         if len(parts) == 2:  # m:ss
#             return float(parts[0]) * 60 + float(parts[1])
#         elif len(parts) == 3:  # h:mm:ss
#             return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
#         else:
#             return float(time_str)


# class DatabaseCompiler:
#     """
#     Compile individual calculation runs into a unified database.
    
#     Directory structure expected:
#     base_dir/
#         data/
#             H2_dA0.8_HF_cc-pVDZ.json
#             H4_dA0.8_DMRG_cc-pVDZ.json
#             ...
#         compute_stats/
#             H2_dA0.8_HF_cc-pVDZ_stats.json
#             H4_dA0.8_DMRG_cc-pVDZ_stats.json
#             ...
#     """
    
#     def __init__(self, base_dir: str, system_name: str = "compiled_system"):
#         """
#         Initialize compiler.
        
#         Args:
#             base_dir: Base directory containing data/ and compute_stats/ subdirectories
#             system_name: Name for the compiled system
#         """
#         self.base_dir = Path(base_dir)
#         self.data_dir = self.base_dir / 'data'
#         self.stats_dir = self.base_dir / 'compute_stats'
#         self.system_name = system_name
#         self.runs = []
        
#         # Verify directory structure
#         if not self.data_dir.exists():
#             raise ValueError(f"Data directory not found: {self.data_dir}")
#         if not self.stats_dir.exists():
#             warnings.warn(f"Compute stats directory not found: {self.stats_dir}")
        
#     def scan_directory(self, pattern: str = "*.json") -> List[Path]:
#         """
#         Scan data directory for result files matching pattern.
        
#         Args:
#             pattern: Glob pattern for JSON filenames (default: "*.json")
            
#         Returns:
#             List of result file paths
#         """
#         result_files = list(self.data_dir.glob(pattern))
#         print(f"Found {len(result_files)} result files in {self.data_dir}")
#         return result_files
    
#     def parse_run_name(self, filename: str) -> Dict:
#         """
#         Parse result filename to extract metadata.
        
#         Expected formats:
#         - H2_dA0.8_HF_cc-pVDZ.json
#         - H4_dA0.8_DMRG_cc-pVDZ.json
#         - N2_R1.09_FCI_aug-cc-pVTZ.json
#         - H_chain_dA0.8_n4_DMRG_cc-pVQZ.json
        
#         Returns:
#             dict: Parsed metadata (geometry_key, method, basis, base_name)
#         """
#         # Remove .json extension
#         base_name = filename.replace('.json', '')
#         parts = base_name.split('_')
        
#         # Try to identify method (HF, FCI, DMRG, CCSDT, MP2)
#         methods = ['HF', 'FCI', 'DMRG', 'CCSDT', 'MP2']
#         method = None
#         method_idx = None
        
#         for i, part in enumerate(parts):
#             if part in methods:
#                 method = part
#                 method_idx = i
#                 break
        
#         if method is None:
#             raise ValueError(f"Could not identify method in filename: {filename}")
        
#         # Basis set is everything after method
#         basis = '_'.join(parts[method_idx + 1:])
        
#         # Geometry is everything before method
#         geometry_parts = parts[:method_idx]
        
#         # Parse geometry to create key
#         geometry_key = self._create_geometry_key(geometry_parts)
        
#         return {
#             'geometry_key': geometry_key,
#             'method': method,
#             'basis': basis,
#             'base_name': base_name,
#             'raw_filename': filename
#         }
    
#     def _create_geometry_key(self, parts: List[str]) -> str:
#         """
#         Create standardized geometry key from parsed parts.
        
#         Examples:
#         - ['H2', 'dA0.8'] -> 'dA_0.80_n2' (assuming default n=2)
#         - ['H4', 'dA0.8'] -> 'dA_0.80_n4'
#         - ['N2', 'R1.09'] -> 'R_1.09'
#         - ['H', 'chain', 'dA0.8', 'n4'] -> 'dA_0.80_n4'
#         """
#         # Look for dA, dB, n, R patterns
#         dA = None
#         dB = None
#         n = None
#         R = None
        
#         # Check for HX or NX pattern (e.g., H2, H4, H6, N2)
#         for part in parts:
#             if part.startswith('H') and len(part) > 1 and part[1:].isdigit():
#                 n = int(part[1:])
#             elif part.startswith('N') and len(part) > 1 and part[1:].isdigit():
#                 n = int(part[1:])
#             elif part.startswith('dA'):
#                 dA = float(part[2:])
#             elif part.startswith('dB'):
#                 dB = float(part[2:])
#             elif part.startswith('n') and part[1:].isdigit():
#                 n = int(part[1:])
#             elif part.startswith('R'):
#                 R = float(part[1:])
        
#         # Build geometry key
#         if R is not None:
#             # Diatomic format
#             return f"R_{R:.2f}"
#         elif dA is not None:
#             # Chain format
#             if n is None:
#                 n = 2  # Default for H2
#             if dB is not None:
#                 return f"dA_{dA:.2f}_dB_{dB:.2f}_n{n}"
#             else:
#                 return f"dA_{dA:.2f}_n{n}"
#         else:
#             # Fallback: join all parts
#             return '_'.join(parts)
    
#     def compile_database(self, pattern: str = "*.json", 
#                         output_file: Optional[str] = None) -> Dict:
#         """
#         Compile all runs into unified database.
        
#         Args:
#             pattern: Glob pattern for result files (default: "*.json")
#             output_file: Optional path to save JSON output
            
#         Returns:
#             dict: Compiled database in CBS analyzer format
#         """
#         result_files = self.scan_directory(pattern)
        
#         # Initialize database structure
#         database = {
#             "system_info": {
#                 "system_name": self.system_name,
#                 "total_runs": 0,
#                 "compilation_timestamp": None
#             }
#         }
        
#         # Track all unique values for system_info
#         all_basis_sets = set()
#         all_methods = set()
#         all_geometries = set()
        
#         successful_runs = 0
#         failed_runs = 0
        
#         for result_file in result_files:
#             try:
#                 # Parse filename
#                 run_info = self.parse_run_name(result_file.name)
#                 geometry_key = run_info['geometry_key']
#                 method = run_info['method']
#                 basis = run_info['basis']
#                 base_name = run_info['base_name']
                
#                 # Load results JSON
#                 with open(result_file, 'r') as f:
#                     results = json.load(f)
                
#                 # Load compute stats (try both .json and .txt extensions, with _stats suffix)
#                 compute_stats = {}
#                 stats_file_json = self.stats_dir / f"{base_name}_stats.json"
#                 stats_file_txt = self.stats_dir / f"{base_name}_stats.txt"
                
#                 if stats_file_json.exists():
#                     # Assume JSON format contains parsed stats
#                     with open(stats_file_json, 'r') as f:
#                         compute_stats = json.load(f)
#                 elif stats_file_txt.exists():
#                     # Parse GNU time output
#                     compute_stats = ComputeStatsParser.parse_time_output(stats_file_txt)
#                 else:
#                     warnings.warn(f"No compute stats found for {base_name}")
                
#                 # Initialize geometry entry if needed
#                 if geometry_key not in database:
#                     database[geometry_key] = {}
                
#                 # Initialize basis entry if needed
#                 if basis not in database[geometry_key]:
#                     database[geometry_key][basis] = {}
                
#                 # Extract method data from results
#                 method_data = self._extract_method_data(results, geometry_key, method, basis)
                
#                 # Merge compute stats
#                 if compute_stats:
#                     # Handle both parsed and raw compute stats formats
#                     wall_time = compute_stats.get('wall_time_seconds') or compute_stats.get('wall_time')
#                     peak_mem = compute_stats.get('max_memory_gb') or compute_stats.get('peak_memory_gb')
                    
#                     if wall_time:
#                         method_data['calculation_time'] = wall_time
#                     if peak_mem:
#                         method_data['peak_memory_gb'] = peak_mem
                    
#                     method_data['compute_stats'] = {
#                         'user_time': compute_stats.get('user_time_seconds') or compute_stats.get('user_time'),
#                         'system_time': compute_stats.get('system_time_seconds') or compute_stats.get('system_time'),
#                         'cpu_percent': compute_stats.get('cpu_percent'),
#                         'page_faults_major': compute_stats.get('page_faults_major'),
#                         'exit_status': compute_stats.get('exit_status')
#                     }
                
#                 # Add to database
#                 database[geometry_key][basis][method] = method_data
#                 # Track metadata
#                 all_basis_sets.add(basis)
#                 all_methods.add(method)
#                 all_geometries.add(geometry_key)
#                 successful_runs += 1
                
#             except Exception as e:
#                 warnings.warn(f"Error processing {result_file.name}: {e}")
#                 failed_runs += 1
#                 continue
        
#         # Update system_info
#         database["system_info"].update({
#             "total_runs": successful_runs,
#             "failed_runs": failed_runs,
#             "basis_sets": sorted(list(all_basis_sets)),
#             "methods": sorted(list(all_methods)),
#             "geometries": sorted(list(all_geometries))
#         })
        
#         print(f"\n{'='*60}")
#         print(f"Compilation Summary")
#         print(f"{'='*60}")
#         print(f"Successful runs: {successful_runs}")
#         print(f"Failed runs: {failed_runs}")
#         print(f"Unique geometries: {len(all_geometries)}")
#         print(f"Unique basis sets: {len(all_basis_sets)}")
#         print(f"Unique methods: {len(all_methods)}")
#         print(f"{'='*60}\n")
        
#         # Save to file if requested
#         if output_file:
#             output_path = Path(output_file)
#             output_path.parent.mkdir(parents=True, exist_ok=True)
#             with open(output_path, 'w') as f:
#                 json.dump(database, f, indent=2)
#             print(f"Database saved to: {output_path}")
        
#         return database
    
#     def _extract_method_data(self, results: Dict, geometry_key: str, 
#                             method: str, basis: str) -> Dict:
#         """
#         Extract method data from results JSON.
        
#         Handles different JSON structures from different run types.
#         Specifically extracts von_neumann_entropy_mo for DMRG calculations.
#         """
#         # Try to find method data in results
#         # Structure could be: results[geometry_key][basis][method]
#         # or results could be the method data directly
        
#         method_data = None
        
#         # Try hierarchical structure
#         if geometry_key in results:
#             if basis in results[geometry_key]:
#                 if method in results[geometry_key][basis]:
#                     method_data = results[geometry_key][basis][method]
        
#         # Try flat structure (results is already method data)
#         if method_data is None and 'method' in results and results.get('method') == method:
#             method_data = results
        
#         # Try looking for method in any geometry key
#         if method_data is None:
#             for geo_key in results.keys():
#                 if geo_key == 'system_info':
#                     continue
#                 if isinstance(results[geo_key], dict) and basis in results[geo_key]:
#                     if method in results[geo_key][basis]:
#                         method_data = results[geo_key][basis][method]
#                         break
        
#         if method_data is None:
#             raise ValueError(f"Could not find method data for {method}/{basis} in results")
        
#         # Make a copy to avoid modifying original
#         extracted_data = method_data.copy()
        
#         # Special handling for DMRG: extract von Neumann entropy
#         if method == 'DMRG' and 'von_neumann_entropy_mo' in method_data:
#             entropy_data = method_data['von_neumann_entropy_mo']
#             if isinstance(entropy_data, dict) and 'entropy' in entropy_data:
#                 # Ensure entropy is at the top level for easy access
#                 extracted_data['von_neumann_entropy'] = entropy_data['entropy']
#                 # Keep the full entropy data structure
#                 extracted_data['von_neumann_entropy_mo'] = entropy_data

#         extracted_data['bond_length'] = results['system_info'].get('d_A_values', None)
        
#         return extracted_data
