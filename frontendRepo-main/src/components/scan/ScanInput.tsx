import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { VideoAnalysisType } from '../../lib/api';
import { useVideoStore } from '../../stores/videoStore';
import styles from './ScanInput.module.css';

export default function ScanInput() {
  const [url, setUrl] = useState('');
  const [analysisType, setAnalysisType] = useState<VideoAnalysisType>('DEEPFAKE');
  const [agreed, setAgreed] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { submitVideoUrl, uploadVideoFile, isSubmitting, error } = useVideoStore();

  const handleScan = async () => {
    if (!agreed || !url.trim() || isSubmitting) return;
    try {
      await submitVideoUrl(url.trim(), analysisType);
      navigate('/scan/analysis');
    } catch {
      // Error message is managed by the video store.
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !agreed || isSubmitting) return;
    try {
      await uploadVideoFile(file, analysisType);
      navigate('/scan/analysis');
    } catch {
      // Error message is managed by the video store.
    }
  };

  return (
    <div className={styles.inputGroup}>
      <div className={styles.inputRow}>
        <input
          type="text"
          placeholder="https://www.example.com/video.mp4"
          aria-label="Video URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
        <input
          ref={fileInputRef}
          className={styles.fileInput}
          type="file"
          accept="video/*"
          onChange={handleFileChange}
        />
        <button
          className={styles.iconBtn}
          aria-label="Upload File"
          type="button"
          disabled={!agreed || isSubmitting}
          onClick={() => fileInputRef.current?.click()}
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </button>
      </div>
      <div className={styles.actionRow}>
        <div className={styles.typeGroup} aria-label="Analysis type">
          <button
            type="button"
            className={`${styles.typeButton} ${analysisType === 'DEEPFAKE' ? styles.typeButtonActive : ''}`}
            onClick={() => setAnalysisType('DEEPFAKE')}
          >
            DEEPFAKE
          </button>
          <button
            type="button"
            className={`${styles.typeButton} ${analysisType === 'T2V' ? styles.typeButtonActive : ''}`}
            onClick={() => setAnalysisType('T2V')}
          >
            T2V
          </button>
        </div>
        <label className={styles.consent}>
          <input
            type="checkbox"
            checked={agreed}
            onChange={(e) => setAgreed(e.target.checked)}
          />
          I agree to terms of analysis
        </label>
        <button
          className={styles.btnPrimary}
          onClick={handleScan}
          disabled={!agreed || !url.trim() || isSubmitting}
        >
          {isSubmitting ? 'REQUESTING' : 'SCAN NOW'} <span className={styles.beta}>BETA</span>
        </button>
      </div>
      {error && <p className={styles.errorText}>{error}</p>}
    </div>
  );
}
