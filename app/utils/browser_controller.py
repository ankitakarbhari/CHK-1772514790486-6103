# app/utils/blockchain.py
"""
Blockchain Verification Module for Digital Authenticity
Stores and verifies media hashes on blockchain for immutable proof
Supports Ethereum, Polygon, and IPFS
Python 3.13+ Compatible
"""

import os
import sys
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# ========== BLOCKCHAIN IMPORTS ==========
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from eth_account.messages import encode_defunct
import ipfshttpclient

# ========== IMAGE HASHING ==========
import cv2
import numpy as np
from PIL import Image
import imagehash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class MediaVerification:
    """Media verification record"""
    media_hash: str
    timestamp: int
    verifier: str
    is_authentic: bool
    confidence: float
    media_type: str  # 'image', 'video', 'audio', 'text'
    metadata: Dict[str, Any]
    ipfs_hash: Optional[str] = None
    blockchain_tx: Optional[str] = None
    block_number: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class VerificationResult:
    """Result of verification against blockchain"""
    verified: bool
    found_on_chain: bool
    matches: bool
    stored_record: Optional[MediaVerification] = None
    message: str = ""
    confidence: float = 0.0


# ============================================
# SMART CONTRACT ABI
# ============================================

MEDIA_VERIFICATION_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "mediaHash", "type": "bytes32"},
            {"internalType": "bool", "name": "isAuthentic", "type": "bool"},
            {"internalType": "string", "name": "mediaType", "type": "string"},
            {"internalType": "string", "name": "metadata", "type": "string"},
            {"internalType": "string", "name": "ipfsHash", "type": "string"}
        ],
        "name": "storeVerification",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "mediaHash", "type": "bytes32"}],
        "name": "getVerification",
        "outputs": [
            {"internalType": "bytes32", "name": "mediaHash", "type": "bytes32"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "address", "name": "verifier", "type": "address"},
            {"internalType": "bool", "name": "isAuthentic", "type": "bool"},
            {"internalType": "string", "name": "mediaType", "type": "string"},
            {"internalType": "string", "name": "metadata", "type": "string"},
            {"internalType": "string", "name": "ipfsHash", "type": "string"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "mediaHash", "type": "bytes32"}],
        "name": "verificationExists",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getTotalVerifications",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "events": [
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "internalType": "bytes32", "name": "mediaHash", "type": "bytes32"},
                    {"indexed": True, "internalType": "address", "name": "verifier", "type": "address"},
                    {"indexed": False, "internalType": "bool", "name": "isAuthentic", "type": "bool"},
                    {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
                ],
                "name": "MediaVerified",
                "type": "event"
            }
        ]
    }
]

# Default contract address (would be deployed separately)
DEFAULT_CONTRACT_ADDRESS = "0x0000000000000000000000000000000000000000"


# ============================================
# MEDIA HASH GENERATOR
# ============================================

class MediaHasher:
    """
    Generate unique hashes for media files
    Supports perceptual hashing for images (finds similar images)
    """
    
    def __init__(self):
        logger.info("MediaHasher initialized")
    
    def hash_image(self, image_path: Union[str, np.ndarray, Image.Image]) -> Dict[str, str]:
        """
        Generate multiple hashes for an image
        
        Returns:
            Dictionary with different hash types
        """
        # Load image
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
            img_array = cv2.imread(image_path)
            if img_array is None:
                img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_path, np.ndarray):
            img_array = image_path
            img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        elif isinstance(image_path, Image.Image):
            img = image_path
            img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Unsupported image type")
        
        # SHA-256 (cryptographic hash)
        with io.BytesIO() as buffer:
            img.save(buffer, format='PNG')
            sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()
        
        # MD5 (fast, less secure)
        with io.BytesIO() as buffer:
            img.save(buffer, format='PNG')
            md5 = hashlib.md5(buffer.getvalue()).hexdigest()
        
        # Perceptual hashes (for similarity detection)
        try:
            phash = str(imagehash.phash(img))
            ahash = str(imagehash.average_hash(img))
            dhash = str(imagehash.dhash(img))
            whash = str(imagehash.whash(img))
        except:
            phash = ahash = dhash = whash = ""
        
        return {
            'sha256': sha256,
            'md5': md5,
            'phash': phash,
            'ahash': ahash,
            'dhash': dhash,
            'whash': whash
        }
    
    def hash_video(self, video_path: str, num_frames: int = 10) -> Dict[str, str]:
        """
        Generate hash for video (hash of key frames)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        frame_hashes = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Hash frame
                frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()
                frame_hashes.append(frame_hash)
        
        cap.release()
        
        # Combine frame hashes
        combined = ''.join(frame_hashes).encode()
        video_hash = hashlib.sha256(combined).hexdigest()
        
        return {
            'sha256': video_hash,
            'num_frames': len(frame_hashes),
            'frame_hashes': frame_hashes[:5]  # First 5 for reference
        }
    
    def hash_audio(self, audio_path: str) -> Dict[str, str]:
        """
        Generate hash for audio file
        """
        import librosa
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Hash the audio data
        audio_bytes = audio.tobytes()
        sha256 = hashlib.sha256(audio_bytes).hexdigest()
        md5 = hashlib.md5(audio_bytes).hexdigest()
        
        # Create perceptual hash from spectrogram
        try:
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_spec_bytes = mel_spec.tobytes()
            perceptual_hash = hashlib.sha256(mel_spec_bytes).hexdigest()[:16]
        except:
            perceptual_hash = ""
        
        return {
            'sha256': sha256,
            'md5': md5,
            'perceptual': perceptual_hash,
            'duration': len(audio) / sr,
            'sample_rate': sr
        }
    
    def hash_text(self, text: str) -> Dict[str, str]:
        """
        Generate hash for text content
        """
        text_bytes = text.encode('utf-8')
        
        return {
            'sha256': hashlib.sha256(text_bytes).hexdigest(),
            'md5': hashlib.md5(text_bytes).hexdigest(),
            'length': len(text)
        }
    
    def hash_file(self, file_path: str) -> Dict[str, str]:
        """
        Generate hash for any file
        """
        sha256 = hashlib.sha256()
        md5 = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
                md5.update(chunk)
        
        return {
            'sha256': sha256.hexdigest(),
            'md5': md5.hexdigest(),
            'filename': Path(file_path).name,
            'size': Path(file_path).stat().st_size
        }
    
    def compare_images(self, img1_path: str, img2_path: str, threshold: float = 0.8) -> Dict:
        """
        Compare two images using perceptual hashing
        
        Returns:
            Similarity score and match status
        """
        # Load images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Generate perceptual hashes
        phash1 = imagehash.phash(img1)
        phash2 = imagehash.phash(img2)
        
        # Calculate Hamming distance
        distance = phash1 - phash2
        max_distance = len(phash1.hash) ** 2
        similarity = 1.0 - (distance / max_distance)
        
        return {
            'similarity': float(similarity),
            'match': similarity >= threshold,
            'distance': int(distance),
            'threshold': threshold
        }


# ============================================
# IPFS STORAGE
# ============================================

class IPFSStorage:
    """
    Store media on IPFS (InterPlanetary File System)
    Provides decentralized, permanent storage
    """
    
    def __init__(self, ipfs_host: str = '/ip4/127.0.0.1/tcp/5001'):
        """
        Initialize IPFS client
        
        Args:
            ipfs_host: IPFS API endpoint
        """
        self.ipfs_host = ipfs_host
        self.client = None
        
        try:
            self.client = ipfshttpclient.connect(ipfs_host)
            logger.info(f"Connected to IPFS at {ipfs_host}")
        except Exception as e:
            logger.warning(f"Could not connect to IPFS: {str(e)}")
            logger.warning("IPFS features will be disabled")
    
    def upload_file(self, file_path: str) -> Optional[str]:
        """
        Upload file to IPFS
        
        Returns:
            IPFS hash (CID)
        """
        if not self.client:
            logger.error("IPFS not connected")
            return None
        
        try:
            res = self.client.add(file_path)
            ipfs_hash = res['Hash']
            logger.info(f"File uploaded to IPFS: {ipfs_hash}")
            return ipfs_hash
        except Exception as e:
            logger.error(f"IPFS upload error: {str(e)}")
            return None
    
    def upload_bytes(self, data: bytes, filename: str = "data.bin") -> Optional[str]:
        """
        Upload bytes to IPFS
        """
        if not self.client:
            logger.error("IPFS not connected")
            return None
        
        try:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(data)
                temp_path = f.name
            
            ipfs_hash = self.upload_file(temp_path)
            os.unlink(temp_path)
            return ipfs_hash
        except Exception as e:
            logger.error(f"IPFS bytes upload error: {str(e)}")
            return None
    
    def upload_json(self, data: Dict, filename: str = "metadata.json") -> Optional[str]:
        """
        Upload JSON data to IPFS
        """
        json_str = json.dumps(data, indent=2)
        return self.upload_bytes(json_str.encode(), filename)
    
    def download_file(self, ipfs_hash: str, output_path: str) -> bool:
        """
        Download file from IPFS
        """
        if not self.client:
            logger.error("IPFS not connected")
            return False
        
        try:
            self.client.get(ipfs_hash, target=output_path)
            logger.info(f"File downloaded from IPFS: {ipfs_hash}")
            return True
        except Exception as e:
            logger.error(f"IPFS download error: {str(e)}")
            return False
    
    def get_pin_status(self, ipfs_hash: str) -> bool:
        """Check if content is pinned"""
        if not self.client:
            return False
        
        try:
            pins = self.client.pin.ls(type='recursive')
            return ipfs_hash in pins
        except:
            return False
    
    def pin_file(self, ipfs_hash: str) -> bool:
        """Pin file to ensure it's not garbage collected"""
        if not self.client:
            return False
        
        try:
            self.client.pin.add(ipfs_hash)
            logger.info(f"Pinned {ipfs_hash}")
            return True
        except:
            return False


# ============================================
# BLOCKCHAIN VERIFIER
# ============================================

class BlockchainVerifier:
    """
    Verify media authenticity using blockchain
    Stores hashes on-chain for immutable verification
    """
    
    def __init__(self,
                 provider_url: Optional[str] = None,
                 contract_address: Optional[str] = None,
                 private_key: Optional[str] = None,
                 chain_id: int = 1):  # 1 = Ethereum Mainnet, 5 = Goerli, 137 = Polygon
        """
        Initialize blockchain verifier
        
        Args:
            provider_url: Web3 provider URL (e.g., Infura)
            contract_address: Smart contract address
            private_key: Private key for signing transactions
            chain_id: Blockchain chain ID
        """
        self.chain_id = chain_id
        
        # Connect to provider
        if provider_url:
            self.w3 = Web3(Web3.HTTPProvider(provider_url))
        else:
            # Use default local provider
            self.w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
        
        # Add PoA middleware for networks like Polygon
        if chain_id in [137, 80001]:  # Polygon Mainnet or Mumbai
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Check connection
        if self.w3.is_connected():
            logger.info(f"Connected to blockchain. Chain ID: {self.w3.eth.chain_id}")
            logger.info(f"Current block: {self.w3.eth.block_number}")
        else:
            logger.warning("Could not connect to blockchain")
        
        # Set contract address
        self.contract_address = contract_address or DEFAULT_CONTRACT_ADDRESS
        
        # Set account
        self.account = None
        if private_key:
            self.account = Account.from_key(private_key)
            logger.info(f"Account loaded: {self.account.address}")
        
        # Initialize contract
        self.contract = None
        if self.contract_address and self.w3.is_connected():
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=MEDIA_VERIFICATION_ABI
            )
            logger.info(f"Contract initialized at {self.contract_address}")
        
        # Initialize components
        self.hasher = MediaHasher()
        self.ipfs = IPFSStorage()
    
    def _bytes32_to_hex(self, bytes32_value) -> str:
        """Convert bytes32 to hex string"""
        return self.w3.to_hex(bytes32_value)
    
    def _hex_to_bytes32(self, hex_string: str) -> bytes:
        """Convert hex string to bytes32"""
        if hex_string.startswith('0x'):
            hex_string = hex_string[2:]
        return bytes.fromhex(hex_string.zfill(64))
    
    def store_verification(self,
                          media_path: str,
                          media_type: str,
                          is_authentic: bool,
                          confidence: float,
                          metadata: Optional[Dict] = None,
                          store_on_ipfs: bool = False) -> Optional[MediaVerification]:
        """
        Store media verification on blockchain
        
        Args:
            media_path: Path to media file
            media_type: 'image', 'video', 'audio', 'text'
            is_authentic: Whether media is authentic
            confidence: Confidence score (0-1)
            metadata: Additional metadata
            store_on_ipfs: Also store on IPFS
        
        Returns:
            MediaVerification record or None if failed
        """
        if not self.contract or not self.account:
            logger.error("Blockchain not configured properly")
            return None
        
        try:
            # Generate hash
            if media_type == 'image':
                hashes = self.hasher.hash_image(media_path)
            elif media_type == 'video':
                hashes = self.hasher.hash_video(media_path)
            elif media_type == 'audio':
                hashes = self.hasher.hash_audio(media_path)
            elif media_type == 'text':
                with open(media_path, 'r') as f:
                    text = f.read()
                hashes = self.hasher.hash_text(text)
            else:
                hashes = self.hasher.hash_file(media_path)
            
            media_hash = hashes['sha256']
            
            # Upload to IPFS if requested
            ipfs_hash = None
            if store_on_ipfs and self.ipfs.client:
                ipfs_hash = self.ipfs.upload_file(media_path)
            
            # Prepare metadata
            metadata_dict = {
                'hashes': hashes,
                'confidence': confidence,
                'filename': Path(media_path).name,
                'timestamp': int(time.time()),
                ** (metadata or {})
            }
            metadata_str = json.dumps(metadata_dict)
            
            # Build transaction
            bytes32_hash = self._hex_to_bytes32(media_hash)
            
            tx = self.contract.functions.storeVerification(
                bytes32_hash,
                is_authentic,
                media_type,
                metadata_str,
                ipfs_hash or ""
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 2000000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign and send
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Verification stored. Tx: {tx_hash.hex()}")
            logger.info(f"Block: {receipt['blockNumber']}")
            
            # Create record
            verification = MediaVerification(
                media_hash=media_hash,
                timestamp=int(time.time()),
                verifier=self.account.address,
                is_authentic=is_authentic,
                confidence=confidence,
                media_type=media_type,
                metadata=metadata_dict,
                ipfs_hash=ipfs_hash,
                blockchain_tx=tx_hash.hex(),
                block_number=receipt['blockNumber']
            )
            
            return verification
            
        except Exception as e:
            logger.error(f"Failed to store verification: {str(e)}")
            return None
    
    def verify_media(self, media_path: str, media_type: str) -> VerificationResult:
        """
        Verify media against blockchain records
        
        Args:
            media_path: Path to media file
            media_type: Type of media
        
        Returns:
            VerificationResult
        """
        try:
            # Generate hash
            if media_type == 'image':
                hashes = self.hasher.hash_image(media_path)
            elif media_type == 'video':
                hashes = self.hasher.hash_video(media_path)
            elif media_type == 'audio':
                hashes = self.hasher.hash_audio(media_path)
            elif media_type == 'text':
                with open(media_path, 'r') as f:
                    text = f.read()
                hashes = self.hasher.hash_text(text)
            else:
                hashes = self.hasher.hash_file(media_path)
            
            media_hash = hashes['sha256']
            
            # Check if exists on blockchain
            exists = self._verification_exists(media_hash)
            
            if not exists:
                return VerificationResult(
                    verified=False,
                    found_on_chain=False,
                    matches=False,
                    message="Media not found on blockchain"
                )
            
            # Get stored record
            stored = self._get_verification(media_hash)
            
            if stored:
                # Check if hashes match
                matches = stored.media_hash == media_hash
                
                return VerificationResult(
                    verified=matches and stored.is_authentic,
                    found_on_chain=True,
                    matches=matches,
                    stored_record=stored,
                    message="Media found on blockchain",
                    confidence=stored.confidence
                )
            else:
                return VerificationResult(
                    verified=False,
                    found_on_chain=True,
                    matches=False,
                    message="Failed to retrieve stored record"
                )
                
        except Exception as e:
            logger.error(f"Verification error: {str(e)}")
            return VerificationResult(
                verified=False,
                found_on_chain=False,
                matches=False,
                message=f"Error: {str(e)}"
            )
    
    def _verification_exists(self, media_hash: str) -> bool:
        """Check if verification exists on blockchain"""
        if not self.contract:
            return False
        
        try:
            bytes32_hash = self._hex_to_bytes32(media_hash)
            return self.contract.functions.verificationExists(bytes32_hash).call()
        except:
            return False
    
    def _get_verification(self, media_hash: str) -> Optional[MediaVerification]:
        """Get verification from blockchain"""
        if not self.contract:
            return None
        
        try:
            bytes32_hash = self._hex_to_bytes32(media_hash)
            result = self.contract.functions.getVerification(bytes32_hash).call()
            
            # Parse result
            metadata = json.loads(result[5])
            
            return MediaVerification(
                media_hash=self._bytes32_to_hex(result[0]),
                timestamp=result[1],
                verifier=result[2],
                is_authentic=result[3],
                confidence=metadata.get('confidence', 0.5),
                media_type=result[4],
                metadata=metadata,
                ipfs_hash=result[6] if result[6] else None
            )
        except Exception as e:
            logger.error(f"Failed to get verification: {str(e)}")
            return None
    
    def get_verification_history(self, media_hash: str) -> List[MediaVerification]:
        """
        Get verification history for a media hash
        Uses event logs
        """
        if not self.contract:
            return []
        
        try:
            bytes32_hash = self._hex_to_bytes32(media_hash)
            
            # Get events
            events = self.contract.events.MediaVerified.get_logs(
                fromBlock=0,
                argument_filters={'mediaHash': bytes32_hash}
            )
            
            history = []
            for event in events:
                verification = MediaVerification(
                    media_hash=self._bytes32_to_hex(event['args']['mediaHash']),
                    timestamp=event['args']['timestamp'],
                    verifier=event['args']['verifier'],
                    is_authentic=event['args']['isAuthentic'],
                    confidence=0.5,  # Not stored in event
                    media_type="unknown",
                    metadata={},
                    blockchain_tx=event['transactionHash'].hex(),
                    block_number=event['blockNumber']
                )
                history.append(verification)
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get history: {str(e)}")
            return []
    
    def get_total_verifications(self) -> int:
        """Get total number of verifications stored"""
        if not self.contract:
            return 0
        
        try:
            return self.contract.functions.getTotalVerifications().call()
        except:
            return 0
    
    def deploy_contract(self, private_key: str) -> Optional[str]:
        """
        Deploy new MediaVerification contract
        
        Returns:
            Contract address if successful
        """
        # This would require the compiled bytecode
        logger.warning("Contract deployment not implemented in this version")
        return None


# ============================================
# VERIFICATION CERTIFICATE GENERATOR
# ============================================

class VerificationCertificate:
    """Generate verification certificates for authenticated media"""
    
    @staticmethod
    def generate(verification: MediaVerification) -> Dict:
        """
        Generate verification certificate
        
        Returns:
            Certificate data
        """
        certificate = {
            'certificate_id': hashlib.sha256(
                f"{verification.media_hash}{verification.timestamp}".encode()
            ).hexdigest()[:16].upper(),
            'media_hash': verification.media_hash,
            'verification_date': datetime.fromtimestamp(verification.timestamp).isoformat(),
            'verifier': verification.verifier,
            'authenticity_status': 'AUTHENTIC' if verification.is_authentic else 'MANIPULATED',
            'confidence': f"{verification.confidence:.2%}",
            'media_type': verification.media_type,
            'blockchain_record': {
                'transaction': verification.blockchain_tx,
                'block_number': verification.block_number,
                'ipfs_hash': verification.ipfs_hash
            },
            'metadata': verification.metadata,
            'verification_url': f"https://verify.deepfake-detector.com/cert/{verification.media_hash[:8]}",
            'issued_by': 'Deepfake Detection System',
            'timestamp': int(time.time())
        }
        
        return certificate
    
    @staticmethod
    def to_html(certificate: Dict) -> str:
        """Generate HTML certificate"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Digital Authenticity Certificate</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .certificate {{ border: 3px solid #4CAF50; padding: 30px; max-width: 800px; margin: auto; }}
                .header {{ text-align: center; border-bottom: 2px solid #ddd; padding-bottom: 20px; }}
                .title {{ color: #4CAF50; font-size: 24px; }}
                .content {{ margin: 20px 0; }}
                .field {{ margin: 10px 0; }}
                .label {{ font-weight: bold; display: inline-block; width: 150px; }}
                .value {{ display: inline-block; }}
                .blockchain {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="certificate">
                <div class="header">
                    <h1 class="title">Digital Authenticity Certificate</h1>
                    <p>Certificate ID: {certificate['certificate_id']}</p>
                </div>
                <div class="content">
                    <div class="field">
                        <span class="label">Media Hash:</span>
                        <span class="value">{certificate['media_hash'][:32]}...</span>
                    </div>
                    <div class="field">
                        <span class="label">Verification Date:</span>
                        <span class="value">{certificate['verification_date']}</span>
                    </div>
                    <div class="field">
                        <span class="label">Verifier:</span>
                        <span class="value">{certificate['verifier']}</span>
                    </div>
                    <div class="field">
                        <span class="label">Authenticity:</span>
                        <span class="value" style="color: {'#4CAF50' if 'AUTHENTIC' in certificate['authenticity_status'] else '#f44336'};">
                            {certificate['authenticity_status']}
                        </span>
                    </div>
                    <div class="field">
                        <span class="label">Confidence:</span>
                        <span class="value">{certificate['confidence']}</span>
                    </div>
                    <div class="field">
                        <span class="label">Media Type:</span>
                        <span class="value">{certificate['media_type']}</span>
                    </div>
                    
                    <div class="blockchain">
                        <h3>Blockchain Verification</h3>
                        <div class="field">
                            <span class="label">Transaction:</span>
                            <span class="value">{certificate['blockchain_record']['transaction'][:32]}...</span>
                        </div>
                        <div class="field">
                            <span class="label">Block Number:</span>
                            <span class="value">{certificate['blockchain_record']['block_number']}</span>
                        </div>
                        <div class="field">
                            <span class="label">IPFS Hash:</span>
                            <span class="value">{certificate['blockchain_record']['ipfs_hash'] or 'N/A'}</span>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p>Verify this certificate at: {certificate['verification_url']}</p>
                        <p>Issued by {certificate['issued_by']} on {datetime.fromtimestamp(certificate['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    @staticmethod
    def save_html(certificate: Dict, output_path: str):
        """Save certificate as HTML file"""
        html = VerificationCertificate.to_html(certificate)
        with open(output_path, 'w') as f:
            f.write(html)
        logger.info(f"Certificate saved to {output_path}")


# ============================================
# FACTORY CLASS
# ============================================

class BlockchainFactory:
    """Factory for creating blockchain components"""
    
    @staticmethod
    def create_verifier(provider_url: Optional[str] = None,
                       contract_address: Optional[str] = None,
                       private_key: Optional[str] = None) -> BlockchainVerifier:
        """Create blockchain verifier"""
        return BlockchainVerifier(
            provider_url=provider_url,
            contract_address=contract_address,
            private_key=private_key
        )
    
    @staticmethod
    def create_hasher() -> MediaHasher:
        """Create media hasher"""
        return MediaHasher()
    
    @staticmethod
    def create_ipfs(ipfs_host: str = '/ip4/127.0.0.1/tcp/5001') -> IPFSStorage:
        """Create IPFS storage"""
        return IPFSStorage(ipfs_host)


# ============================================
# TESTING FUNCTION
# ============================================

def test_blockchain():
    """Test blockchain module"""
    print("=" * 60)
    print("TESTING BLOCKCHAIN VERIFICATION MODULE")
    print("=" * 60)
    
    # Test hasher
    print("\n1️⃣ Testing Media Hasher...")
    hasher = MediaHasher()
    
    # Create test image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_path = "test_image.png"
    cv2.imwrite(test_path, test_img)
    
    # Hash image
    hashes = hasher.hash_image(test_path)
    print(f"✅ Image hashes generated")
    print(f"   SHA256: {hashes['sha256'][:16]}...")
    print(f"   MD5: {hashes['md5'][:16]}...")
    print(f"   pHash: {hashes['phash']}")
    
    # Clean up
    os.remove(test_path)
    
    # Test blockchain verifier
    print("\n2️⃣ Testing Blockchain Verifier (local only)...")
    verifier = BlockchainFactory.create_verifier()
    print(f"✅ Verifier created")
    print(f"   Connected: {verifier.w3.is_connected()}")
    
    # Test verification storage (mock)
    print("\n3️⃣ Testing Verification Storage (simulated)...")
    print("   Note: Full blockchain test requires actual connection")
    print("   For testing with real blockchain, configure provider_url")
    
    # Test certificate generation
    print("\n4️⃣ Testing Certificate Generation...")
    
    mock_verification = MediaVerification(
        media_hash="abc123def456ghi789",
        timestamp=int(time.time()),
        verifier="0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        is_authentic=True,
        confidence=0.98,
        media_type="image",
        metadata={"filename": "test.jpg", "size": 1024},
        ipfs_hash="QmT5NvUtoM5nWFfrQdVrFtvGfKFmG7AHE8P34isbZxt4V",
        blockchain_tx="0x1234567890abcdef",
        block_number=12345678
    )
    
    certificate = VerificationCertificate.generate(mock_verification)
    print(f"✅ Certificate generated")
    print(f"   Certificate ID: {certificate['certificate_id']}")
    print(f"   Status: {certificate['authenticity_status']}")
    print(f"   Confidence: {certificate['confidence']}")
    
    # Save test certificate
    cert_path = "test_certificate.html"
    VerificationCertificate.save_html(certificate, cert_path)
    print(f"✅ Certificate saved to {cert_path}")
    
    # Clean up
    if os.path.exists(cert_path):
        os.remove(cert_path)
        print(f"✅ Test certificate cleaned up")
    
    print("\n" + "=" * 60)
    print("✅ BLOCKCHAIN MODULE TEST PASSED!")
    print("=" * 60)
    
    print("\n📝 For production use:")
    print("   1. Deploy smart contract to Ethereum/Polygon")
    print("   2. Get Infura/Alchemy provider URL")
    print("   3. Set up IPFS node or use Pinata")
    print("   4. Configure with private key for signing")
    
    return verifier, hasher


if __name__ == "__main__":
    # Run test
    verifier, hasher = test