"""Generate professional INRC proposal PDF using fpdf2."""
from fpdf import FPDF
import os

class ProposalPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, 'INRC Industry Member Proposal - NeuroCUDA / Quantaracore Technologies LLP', align='C')
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(20, 60, 120)
        self.cell(0, 8, title)
        self.ln(10)

    def sub_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 7, title)
        self.ln(8)

    def body(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(5)
        self.cell(3, 5.5, '-')
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def simple_table(self, headers, rows, col_widths=None):
        if not col_widths:
            col_widths = [190 / len(headers)] * len(headers)
        # Header
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, ' ' + h, border=0, fill=True)
        self.ln()
        # Rows
        self.set_font('Helvetica', '', 9)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(240, 244, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, ' ' + str(cell), border=0, fill=True)
            self.ln()
            fill = not fill
        self.ln(4)


def build_pdf():
    pdf = ProposalPDF('P', 'mm', 'A4')
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, 20)
    pdf.add_page()

    # ============ COVER PAGE ============
    pdf.ln(30)
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_text_color(20, 60, 120)
    pdf.multi_cell(0, 12, 'INRC Industry Member\nProposal')
    pdf.ln(6)
    pdf.set_font('Helvetica', '', 16)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, 'NeuroCUDA: Multi-Backend SNN Compiler')
    pdf.ln(10)
    pdf.cell(0, 8, 'with Loihi 2 Hardware Validation')
    pdf.ln(16)
    pdf.set_draw_color(20, 60, 120)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(12)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(80, 80, 80)
    info = [
        'Submitted to: Intel Neuromorphic Research Community (INRC)',
        'Submitted by: Quantaracore Technologies LLP',
        'Date: June 28, 2026',
        'Contact: Krishna Varma, Founder',
        'Email: founder@quantaracore.in',
        'Website: https://quantaracore.in/neurocuda',
        'Member Type: Industry Member',
        'Research Vectors: RV2 (Algorithms) + RV3 (Applications)',
    ]
    for line in info:
        pdf.cell(0, 7, line)
        pdf.ln(7)

    # ============ SECTION 1: PARTICIPANTS ============
    pdf.add_page()
    pdf.section_title('1. Participants')
    pdf.simple_table(
        ['Name', 'Role', 'Affiliation', 'Email'],
        [['Krishna Varma', 'Principal Investigator / Founder', 'Quantaracore Technologies LLP', 'founder@quantaracore.in']],
        [38, 42, 62, 48]
    )

    # ============ SECTION 2: ABSTRACT ============
    pdf.section_title('2. Project Abstract')
    pdf.body(
        'NeuroCUDA is an open-source compiler that converts trained PyTorch models to spiking '
        'neural networks and deploys them across GPU, CPU, BrainScaleS-2 analog silicon (Heidelberg), '
        'and SpiNNaker-1 digital silicon (Manchester) through one Python API call. We maintain a '
        'Loihi 2 simulator backend validated against published Loihi neuron equations (0 spike '
        'deviations per 100,000+ test vectors) but have never validated against physical Loihi 2 hardware.'
    )
    pdf.body(
        'This project closes that gap: deploy converted SNNs to Loihi 2 cloud hardware, validate '
        'bit-accurate spike output against our simulator, measure real energy consumption using Intel '
        'Lava energy probes, and produce honest multi-backend NeuroBench benchmarks comparing three '
        'distinct physical neuromorphic chips (Loihi 2, SpiNNaker-1, BrainScaleS-2) from one unified '
        'compiler pipeline. All Loihi 2 integration code will be released as MIT-licensed open-source.'
    )

    # ============ SECTION 3: PROJECT DESCRIPTION ============
    pdf.section_title('3. Project Description')

    pdf.sub_title('3.1 The Problem')
    pdf.body(
        'Spiking neural network research suffers from platform fragmentation. Each neuromorphic '
        'hardware system (Loihi 2, SpiNNaker-1, BrainScaleS-2) requires its own SDK, model format, '
        'and benchmarking methodology. No open-source compiler takes one PyTorch model and deploys '
        'it to multiple physical neuromorphic chips with honest, reproducible benchmarks.'
    )

    pdf.sub_title('3.2 What We Already Built')
    pdf.simple_table(
        ['Backend', 'Type', 'Status'],
        [
            ['GPU (PyTorch)', 'Simulator', 'Production. CUDA-accelerated.'],
            ['CPU (PyTorch)', 'Simulator', 'Bit-exact: 0/256K spike devs vs GPU'],
            ['Loihi 2 IF model', 'Simulator', '0/100K+ spike devs vs published Loihi equations'],
            ['SpiNNaker-1', 'Physical silicon', '269K-param MLP compiles. 5000 core-hr EBRAINS quota approved'],
            ['BrainScaleS-2', 'Physical silicon', '138-neuron SNN on chip 57 (Heidelberg). Confirmed 2026-06-28'],
            ['FPGA', 'HLS C++', 'NIR export pipeline working. Not yet synthesized'],
        ],
        [40, 40, 110]
    )

    pdf.sub_title('3.3 What is Missing - Physical Loihi 2')
    pdf.body(
        'Our Loihi 2 simulator implements the published neuron equations correctly (0 spike '
        'deviations validated) but has never been compared to physical Loihi 2 silicon output, '
        'cannot measure real energy consumption (uses datasheet estimates: 0.08 pJ/SynOp), '
        'cannot account for fabrication variation or on-chip noise, and cannot honestly claim '
        'physical silicon per the NeuroBench hardware labeling standard.'
    )

    pdf.sub_title('3.4 Approach')
    pdf.body(
        'Trained PyTorch ANN -> Stage 1: QCFS Calibration (learn per-channel thresholds, 5 epochs) '
        '-> Stage 2: IF Neuron Replacement + BPTT Fine-Tuning (binary spikes, surrogate gradients, '
        '5 epochs) -> NIR Export (vendor-neutral graph format) -> Lava NIR Importer -> Loihi 2 Cloud '
        'Hardware (vLab) -> Validation: Spike comparison + Energy measurement + NeuroBench report.'
    )
    pdf.body('Validation sequence (ascending complexity):')
    pdf.bullet('1. Single IF neuron: validate bit-identical spike times (simulator vs hardware)')
    pdf.bullet('2. MLP MNIST (784-256-256-10, 269K params, 97.4% SNN): full network, measure accuracy')
    pdf.bullet('3. CNN N-MNIST (event-camera, 99.88% SNN, 92% sparse): temporal input, stress-test pipeline')
    pdf.bullet('4. Multi-seed NeuroBench report: 3 seeds, full test sets, real energy from Lava probes')

    pdf.sub_title('3.5 Comparison to Prior Work')
    pdf.simple_table(
        ['Tool', 'Multi-Chip', 'Physical Silicon', 'NeuroBench', 'License'],
        [
            ['SNNToolBox', 'Partial', 'Partial', 'No', 'Yes'],
            ['Lava-DL', 'Loihi only', 'Yes (Loihi 2)', 'No', 'Yes'],
            ['sPyNNaker', 'SpiNNaker only', 'Yes', 'No', 'Yes'],
            ['NIR', 'Format only', 'No', 'No', 'Yes'],
            ['NeuroCUDA', '3 chips', '3 chips', 'Yes', 'MIT'],
        ],
        [33, 33, 40, 35, 30]
    )

    pdf.sub_title('3.6 Unique Value of Loihi 2')
    pdf.bullet('Digital deterministic architecture: bit-reproducible results, unlike analog BSS-2')
    pdf.bullet('Lava SDK with NIR support: clean Python API, no manual PyNN scripting needed')
    pdf.bullet('Per-synapse weight programming: arbitrary matrices, unlike BSS-2 (masks only)')
    pdf.bullet('Built-in energy measurement: Lava probes provide real pJ data, not datasheet estimates')
    pdf.bullet('8-bit signed weights: matches NeuroCUDA per-channel quantization pipeline exactly')

    pdf.sub_title('3.7 Quantitative Evaluation')
    pdf.simple_table(
        ['Metric', 'Target', 'Method'],
        [
            ['Spike deviation', '0 spike time deviations', 'Simulator vs hardware, identical input'],
            ['Accuracy gap (MLP MNIST)', '<=2% (97.4% -> >=95.4%)', 'Full 10K test set, Loihi 2 vs simulator'],
            ['Accuracy gap (CNN NMNIST)', '<=2% (99.88% -> >=97.88%)', 'Full 10K test set, Loihi 2 vs simulator'],
            ['Real energy vs estimate', 'Within 2x of 0.08 pJ/SynOp', 'Lava energy probes'],
            ['NeuroBench report', 'Complete 3-chip table', 'NeuroBench standard, Nature Comms 2025'],
        ],
        [52, 62, 76]
    )

    pdf.sub_title('3.8 Definition of Success')
    pdf.body(
        'A published, reproducible NeuroBench comparison table with verified numbers from three '
        'distinct physical neuromorphic chips (Loihi 2, SpiNNaker-1, BrainScaleS-2) measured by '
        'one unified open-source compiler pipeline. No other tool currently provides this.'
    )

    pdf.sub_title('3.9 Citations')
    cites = [
        'Davies, M. et al. "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning." IEEE Micro, 2018.',
        'Orchard, G. et al. "Efficient Neuromorphic Signal Processing with Loihi 2." IEEE ISCAS, 2021.',
        'Lava Software Framework. https://github.com/lava-nc/lava',
        'NIR - Neuromorphic Intermediate Representation. https://github.com/neuromorphs/NIR',
        'Yik, J. et al. "NeuroBench: Benchmarking Neuromorphic Computing." Nature Communications, 2025.',
        'Bu, T. et al. "Optimal Quantization for SNNs via Calibrated Floor-Shift." NeurIPS, 2023.',
    ]
    for i, c in enumerate(cites, 1):
        pdf.bullet(f'[{i}] {c}')

    # ============ SECTION 4: RESEARCH PLAN ============
    pdf.add_page()
    pdf.section_title('4. Research Plan')

    pdf.sub_title('4.1 Deliverables')
    pdf.simple_table(
        ['#', 'Deliverable', 'Type', 'License'],
        [
            ['D1', 'Single-neuron Loihi 2 validation report', 'Technical report', 'Public'],
            ['D2', 'Lava-based Loihi 2 backend module', 'Python module', 'MIT'],
            ['D3', 'MLP MNIST + CNN NMNIST deployment code', 'Python module', 'MIT'],
            ['D4', 'Multi-backend NeuroBench report (3 chips)', 'Public benchmark', 'CC-BY'],
            ['D5', 'Tutorial: Deploy PyTorch to Loihi 2 in One Line', 'Blog + Seminar', 'Public'],
        ],
        [10, 82, 48, 30]
    )

    pdf.sub_title('4.2 Personnel')
    pdf.body(
        'Krishna Varma - Principal Investigator - Compiler pipeline, Lava integration, benchmark '
        'execution, NeuroBench reporting, documentation. Quantaracore Technologies LLP.'
    )

    pdf.sub_title('4.3 Milestones')
    pdf.simple_table(
        ['Week', 'Milestone', 'Deliverable'],
        [
            ['1', 'INRC onboarding. Lava configured. Cloud access confirmed.', 'Setup verified'],
            ['2', 'Single IF neuron on Loihi 2. Spike output validated vs simulator.', 'D1 complete'],
            ['3-4', 'MLP MNIST deployed via NIR->Lava. Accuracy measured on hardware.', 'D2, D3 partial'],
            ['5-6', 'CNN NMNIST deployed. Multi-seed benchmarks. Energy measured.', 'D3 done, D4 partial'],
            ['7-8', 'Lava backend complete. nc.compile(target=loihi2_lava) working.', 'D2 complete'],
            ['9-10', 'NeuroBench report finalized. INRC seminar presented.', 'D4 complete'],
            ['11-12', 'Tutorial published. Code merged to main. Public release.', 'D5 complete'],
        ],
        [18, 95, 77]
    )

    pdf.sub_title('4.4 Technical Tradeoffs')
    pdf.bullet('Quantization: Use NeuroCUDA per-channel 8-bit quantization (matches Loihi 2 native format)')
    pdf.bullet('Partitioning: MLP MNIST fits single-chip. CNN NMNIST may need 2-chip via Lava multi-chip compiler')
    pdf.bullet('Encoding: Rate coding (Poisson) for MNIST. Event-driven for NMNIST. Both supported by Lava')

    # ============ SECTION 5: RESOURCE NEEDS ============
    pdf.section_title('5. Loihi Resource Needs')
    pdf.body('Project specifically targets Loihi 2 capabilities. No on-site hardware required.')

    pdf.simple_table(
        ['Model', 'Params', 'Synapses', 'Chips'],
        [
            ['Single-neuron validation', '<10', '<10', '<1'],
            ['MLP MNIST (3-layer)', '269,322', '~269,000', '1'],
            ['CNN NMNIST (3-layer Conv)', '147,466', '~1,500,000', '1-2'],
        ],
        [55, 35, 50, 40]
    )

    pdf.body(
        'Cloud access pattern: Occasional interactive sessions (2-3 per week, 1-2 hours each). '
        'Batch inference runs for benchmark collection (10-20 runs per session, ~100ms each). '
        'Total estimated: approximately 100 core-hours over 12 weeks. Single-system, occasional '
        'access is sufficient.'
    )
    pdf.body(
        'Justification: Our simulator implements neuron equations correctly but cannot model '
        'fabrication variation, on-chip noise, or real system-level energy. Physical Loihi 2 '
        'access is required for honest physical silicon labeling per NeuroBench standard.'
    )

    # ============ SECTION 6: DELIVERABLES ============
    pdf.add_page()
    pdf.section_title('6. Material Deliverables to INRC')

    pdf.sub_title('6.1 Software Contributions (MIT License)')
    pdf.bullet('neurocuda/backends/loihi2_lava.py - Lava-based Loihi 2 physical silicon backend')
    pdf.bullet('NIR->Lava bridge improvements (upstreamed to lava-nc if gaps found during validation)')
    pdf.bullet('Energy measurement harness (standardize Lava probe output -> NeuroBench JSON format)')
    pdf.bullet('Benchmark scripts and raw data (all publicly reproducible)')

    pdf.sub_title('6.2 Documentation and Knowledge Sharing')
    pdf.bullet('Deploy PyTorch to Loihi 2 in One API Call - blog post + Jupyter notebook tutorial')
    pdf.bullet('Multi-chip NeuroBench report - markdown table + JSON export (3 physical chips)')
    pdf.bullet('INRC Forum seminar - 30-minute presentation + Q&A')
    pdf.bullet('Loihi 2 validation technical report - PDF with spike traces and energy logs')

    pdf.sub_title('6.3 Datasets and Benchmarks')
    pdf.body(
        'All benchmark data publicly released: Loihi 2 spike output traces (simulator vs hardware, '
        'per-neuron, per-timestep), energy measurement logs (Lava probe output per inference), '
        'NeuroBench-format JSON reports (accuracy, latency, energy, sparsity), and comparison '
        'tables across all three physical chips.'
    )

    # ============ SECTION 7: IP ============
    pdf.section_title('7. Intellectual Property')

    pdf.sub_title('7.1 Background IP')
    pdf.body(
        'NeuroCUDA is MIT-licensed open-source software. All existing code (GPU, CPU, Loihi 2 '
        'simulator, SpiNNaker-1, BrainScaleS-2 backends; QCFS converter; NIR exporter; NeuroBench '
        'reporter) is publicly available under the MIT license at https://quantaracore.in/neurocuda. '
        'No background IP restrictions.'
    )

    pdf.sub_title('7.2 IP Created Under This Project')
    pdf.body(
        'All software, documentation, and benchmark data created under this project will be released '
        'as MIT-licensed open-source to the public domain. Quantaracore Technologies LLP does not '
        'seek to retain proprietary rights over INRC project outputs.'
    )

    pdf.sub_title('7.3 Agreement Type')
    pdf.body(
        'Corporate Participation Agreement requested. Please send to founder@quantaracore.in '
        'for review and execution by Quantaracore Technologies LLP.'
    )

    pdf.sub_title('7.4 Funding')
    pdf.body('No Intel funding is requested. We request Loihi 2 cloud hardware access for validation only.')

    # ============ SECTION 8: SUBMISSION ============
    pdf.section_title('8. Submission Details')
    pdf.simple_table(
        ['Field', 'Value'],
        [
            ['Name', 'Krishna Varma'],
            ['Title', 'Founder'],
            ['Company', 'Quantaracore Technologies LLP'],
            ['Email', 'founder@quantaracore.in'],
            ['Website', 'https://quantaracore.in/neurocuda'],
            ['Member Type', 'Industry Member'],
            ['Research Vectors', 'RV2 (Algorithms) + RV3 (Applications)'],
        ],
        [48, 142]
    )
    pdf.ln(4)
    pdf.body(
        'Submitted via email to inrc_interest@intel.com (online Qualtrics form reported inactive). '
        'Supporting materials: NeuroCUDA GitHub repository, multi-backend validation (5 backends, '
        '3 simulators + 2 physical silicon confirmed), BrainScaleS-2 chip 57 confirmation (June 28, '
        '2026), SpiNNaker-1 EBRAINS quota (5000 core-hours), NeuroBench standard compliance.'
    )
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, 'Submitted June 28, 2026 by Krishna Varma on behalf of Quantaracore Technologies LLP.', align='C')

    # ============ SAVE ============
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'INRC_Proposal_NeuroCUDA_Quantaracore.pdf')
    pdf.output(output_path)
    print(f'PDF generated: {output_path}')
    print(f'Pages: {pdf.page_no()}')
    size_kb = os.path.getsize(output_path) / 1024
    print(f'Size: {size_kb:.0f} KB')

if __name__ == '__main__':
    build_pdf()
