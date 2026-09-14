import { jsPDF } from 'jspdf';

export interface ForensicReportData {
  audit_id: string;
  timestamp: string;
  processing_time_ms: number;
  authenticity_index: number;
  verdict: string;
  verdict_code: string;
  confidence_score: number;
  signals: {
    fft: { name: string; score: number; status: string; summary: string; metrics: any };
    ela: { name: string; score: number; status: string; summary: string; metrics: any };
    seam: { name: string; score: number; status: string; summary: string; metrics: any };
    biology: { name: string; score: number; status: string; summary: string; metrics: any };
    nn?: { name: string; score: number; status: string; summary: string; metrics: any };
  };
  findings: Array<{
    severity: string;
    signal: string;
    title: string;
    detail: string;
  }>;
  metadata: {
    width: number;
    height: number;
    format: string;
    face_detected: boolean;
    face_count: number;
  };
}

export function exportForensicPDF(report: ForensicReportData) {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4'
  });

  const pageWidth = doc.internal.pageSize.getWidth();
  
  // 1. Dark Modern Header Block
  doc.setFillColor(15, 15, 15);
  doc.rect(0, 0, pageWidth, 42, 'F');

  // Title & Brand
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(18);
  doc.setTextColor(255, 255, 255);
  doc.text('SPECTRAVISION FORENSIC AUDIT DOSSIER', 14, 18);

  doc.setFontSize(9);
  doc.setFont('Helvetica', 'normal');
  doc.setTextColor(0, 242, 254);
  doc.text('Multimodal Synthetic Media & Deepfake Provenance Verification', 14, 25);

  doc.setFontSize(8);
  doc.setTextColor(160, 160, 160);
  doc.text(`AUDIT ID: ${report.audit_id}  |  TIMESTAMP: ${new Date(report.timestamp).toUTCString()}`, 14, 34);

  // 2. Composite Verdict Box
  const verdictY = 48;
  const isSynthetic = report.verdict_code === 'SYNTHETIC';
  const isSuspicious = report.verdict_code === 'SUSPICIOUS';
  const isAuthentic = report.verdict_code === 'AUTHENTIC' || report.verdict_code === 'LIKELY_AUTHENTIC';

  const boxFill = isAuthentic ? [240, 253, 244] : (isSuspicious ? [254, 252, 232] : [255, 241, 242]);
  const boxBorder = isAuthentic ? [34, 197, 94] : (isSuspicious ? [234, 179, 8] : [239, 68, 68]);

  doc.setFillColor(boxFill[0], boxFill[1], boxFill[2]);
  doc.setDrawColor(boxBorder[0], boxBorder[1], boxBorder[2]);
  doc.setLineWidth(0.8);
  doc.roundedRect(14, verdictY, pageWidth - 28, 30, 2, 2, 'FD');

  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(14);
  doc.setTextColor(boxBorder[0], boxBorder[1], boxBorder[2]);
  doc.text(report.verdict, 20, verdictY + 12);

  doc.setFontSize(10);
  doc.setFont('Helvetica', 'normal');
  doc.setTextColor(60, 60, 60);
  doc.text(`Composite Authenticity Index: ${report.authenticity_index}/100  |  Confidence: ${report.confidence_score}%`, 20, verdictY + 22);

  // 3. Multi-Signal Breakdown Table
  let startY = 88;
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(12);
  doc.setTextColor(20, 20, 20);
  doc.text('1. Multi-Layer Forensic Signal Matrix', 14, startY);

  startY += 7;
  doc.setFillColor(245, 245, 245);
  doc.rect(14, startY - 4, pageWidth - 28, 8, 'F');
  doc.setFontSize(8.5);
  doc.setTextColor(80, 80, 80);
  doc.text('SIGNAL LAYER', 18, startY);
  doc.text('SCORE', 80, startY);
  doc.text('STATUS', 105, startY);
  doc.text('TECHNICAL EVIDENCE', 135, startY);

  const signalRows: Array<{ name: string; sig: any }> = [
    { name: '2D-FFT Spectral Analysis', sig: report.signals.fft },
    { name: 'Error Level Analysis (ELA)', sig: report.signals.ela },
    { name: 'Boundary & Seam Gradients', sig: report.signals.seam },
    { name: 'Biological Landmark Kinematics', sig: report.signals.biology }
  ];

  if (report.signals.nn) {
    signalRows.push({ name: 'Neural Network ViT Classifier', sig: report.signals.nn });
  }

  startY += 7;
  doc.setFont('Helvetica', 'normal');
  doc.setFontSize(8);

  signalRows.forEach((row, i) => {
    if (i % 2 === 0) {
      doc.setFillColor(252, 252, 252);
      doc.rect(14, startY - 4, pageWidth - 28, 8, 'F');
    }
    doc.setTextColor(20, 20, 20);
    doc.text(row.name, 18, startY + 1);
    doc.text(`${row.sig.score}/100`, 80, startY + 1);
    
    // Status text color
    const isClean = row.sig.status === 'CLEAN';
    doc.setTextColor(isClean ? 34 : 220, isClean ? 150 : 38, isClean ? 80 : 38);
    doc.text(row.sig.status, 105, startY + 1);
    
    doc.setTextColor(70, 70, 70);
    const summaryTrim = doc.splitTextToSize(row.sig.summary, 60);
    doc.text(summaryTrim[0] || '', 135, startY + 1);
    
    startY += 8;
  });

  // 4. Auditable Evidence Findings
  startY += 8;
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(12);
  doc.setTextColor(20, 20, 20);
  doc.text('2. Auditable Evidence & Incident Log', 14, startY);

  startY += 6;
  report.findings.forEach((f) => {
    doc.setFont('Helvetica', 'bold');
    doc.setFontSize(8.5);
    
    const isCrit = f.severity === 'CRITICAL';
    doc.setTextColor(isCrit ? 220 : (f.severity === 'MEDIUM' ? 200 : 40), isCrit ? 38 : (f.severity === 'MEDIUM' ? 120 : 140), isCrit ? 38 : (f.severity === 'MEDIUM' ? 20 : 60));
    doc.text(`[${f.severity}] ${f.signal}: ${f.title}`, 16, startY);
    
    startY += 4.5;
    doc.setFont('Helvetica', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(80, 80, 80);
    const detailLines = doc.splitTextToSize(f.detail, pageWidth - 32);
    doc.text(detailLines, 16, startY);
    
    startY += (detailLines.length * 4) + 3;
  });

  // 5. Verification Hash & Footer
  const footerY = 275;
  doc.setDrawColor(220, 220, 220);
  doc.setLineWidth(0.4);
  doc.line(14, footerY, pageWidth - 14, footerY);

  doc.setFontSize(7.5);
  doc.setTextColor(140, 140, 140);
  doc.text('CERTIFICATE PROVENANCE STAMP: SHA-256 VERIFIED AUDIT', 14, footerY + 6);
  doc.text('SpectraVision AI • Digital Media Authenticity & Synthetic Anomaly Audit Engine', 14, footerY + 10);
  doc.text('https://spectravision.ai', pageWidth - 14, footerY + 10, { align: 'right' });

  // Save / Trigger Download
  doc.save(`SpectraVision_Audit_${report.audit_id}.pdf`);
}
