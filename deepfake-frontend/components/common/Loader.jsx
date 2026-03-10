import Loader from '@/components/common/Loader';
import { detectImage } from '@/utils/api';
import { useAlert } from '@/components/common/Alert';

const ImageUploader = () => {
  const [loading, setLoading] = useState(false);
  const { error } = useAlert();

  const handleUpload = async (file) => {
    setLoading(true);
    try {
      const result = await detectImage(file);
      console.log('Result:', result);
    } catch (err) {
      error('Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {loading ? (
        <Loader text="Analyzing image..." />
      ) : (
        <button onClick={() => handleUpload(file)}>Upload</button>
      )}
    </div>
  );
};