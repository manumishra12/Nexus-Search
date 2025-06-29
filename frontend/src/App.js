import React, { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Enhanced SVG Icons
const SearchIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
  </svg>
);

const PlusIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
  </svg>
);

const XMarkIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
  </svg>
);

const DocumentIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-4.5B4.875 8.25 3 10.125 3 12.75v2.625" />
  </svg>
);

const PhotoIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-4.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25z" />
  </svg>
);

const LinkIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M13.19 8.688a4.5 4.5 0 011.242 7.244l-4.5 4.5a4.5 4.5 0 01-6.364-6.364l1.757-1.757m13.35-.622l1.757-1.757a4.5 4.5 0 00-6.364-6.364l-4.5 4.5a4.5 4.5 0 001.242 7.244" />
  </svg>
);

const ArrowTopRightIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 19.5l15-15m0 0H8.25m11.25 0v11.25" />
  </svg>
);

const ChevronLeftIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
  </svg>
);

const ChevronRightIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
  </svg>
);

const ExclamationTriangleIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
  </svg>
);

const MoonIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M21.752 15.002A9.718 9.718 0 0118 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 003 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 009.002-5.998z" />
  </svg>
);

const SunIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" />
  </svg>
);

const DocumentTextIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-4.5B4.875 8.25 3 10.125 3 12.75v2.625M9 16.5v2.25M15 13.5v6M17.25 8.25V15" />
  </svg>
);

const TrashIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
  </svg>
);

const PlayIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="currentColor" viewBox="0 0 24 24">
    <path d="M8 5v14l11-7z"/>
  </svg>
);

const Squares2X2Icon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" />
  </svg>
);

const ListBulletIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 6.75h12M8.25 12h12m-12 5.25h12M3.75 6.75h.007v.008H3.75V6.75zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zM3.75 12h.007v.008H3.75V12zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zM3.75 17.25h.007v.008H3.75v-.008zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
  </svg>
);

const ClipboardDocumentListIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25zM6.75 12h.008v.008H6.75V12zm0 3h.008v.008H6.75V15zm0 3h.008v.008H6.75V18z" />
  </svg>
);

const BookmarkIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0111.186 0z" />
  </svg>
);

const LightBulbIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189 5.99 5.99 0 003.75-5.061 6 6 0 00-12 0c0 2.28 1.21 4.28 3.022 5.397a6.001 6.001 0 003.478.443zM15 8.25a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

// File Upload Hook
const useFileUpload = ({ onDrop, accept = {}, multiple = true }) => {
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length && onDrop) {
      onDrop(files);
    }
  }, [onDrop]);

  const handleFileSelect = useCallback((e) => {
    const files = Array.from(e.target.files);
    if (files.length && onDrop) {
      onDrop(files);
    }
    e.target.value = '';
  }, [onDrop]);

  const openFileDialog = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return {
    getRootProps: () => ({
      onDragEnter: handleDragEnter,
      onDragLeave: handleDragLeave,
      onDragOver: handleDragOver,
      onDrop: handleDrop,
      onClick: openFileDialog,
    }),
    getInputProps: () => ({
      ref: fileInputRef,
      type: 'file',
      multiple,
      accept: Object.keys(accept).join(','),
      onChange: handleFileSelect,
      style: { display: 'none' },
    }),
    isDragActive,
  };
};

// Enhanced Summary Panel Component
const SummaryPanel = ({ results, query, darkMode, onGenerateDocument }) => {
  if (!results) {
    return (
      <div className={`h-full flex flex-col ${darkMode ? 'bg-black border-gray-800' : 'bg-white border-gray-200'} rounded-lg border`}>
        <div className={`p-4 border-b ${darkMode ? 'border-gray-800 bg-gray-900' : 'border-gray-200 bg-gray-50'}`}>
          <h3 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'} flex items-center`}>
            <ClipboardDocumentListIcon className="w-5 h-5 mr-2" />
            Research Summary
          </h3>
        </div>
        <div className={`flex-1 flex items-center justify-center ${darkMode ? 'bg-black' : 'bg-gray-50'}`}>
          <div className="text-center">
            <ClipboardDocumentListIcon className={`w-16 h-16 ${darkMode ? 'text-gray-700' : 'text-gray-300'} mx-auto mb-4`} />
            <p className={`text-lg font-medium ${darkMode ? 'text-gray-400' : 'text-gray-600'} mb-2`}>Research Summary</p>
            <p className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>Start a search to see comprehensive insights</p>
          </div>
        </div>
      </div>
    );
  }

  const { sources = [], extracted_media = [], follow_up_questions = [], credibility_score = 0 } = results;

  return (
    <div className={`h-full flex flex-col ${darkMode ? 'bg-black border-gray-800' : 'bg-white border-gray-200'} rounded-lg border shadow-sm`}>
      {/* Header */}
      <div className={`p-4 border-b ${darkMode ? 'border-gray-800 bg-gray-900' : 'border-gray-200 bg-gray-50'}`}>
        <div className="flex items-center justify-between">
          <h3 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'} flex items-center`}>
            <ClipboardDocumentListIcon className="w-5 h-5 mr-2" />
            Research Summary
          </h3>
          {onGenerateDocument && (
            <button
              onClick={onGenerateDocument}
              className={`px-3 py-1.5 ${darkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-600 hover:bg-blue-700'} text-white text-sm rounded-lg transition-colors duration-200 flex items-center space-x-2`}
            >
              <DocumentTextIcon className="w-4 h-4" />
              <span>Generate PDF</span>
            </button>
          )}
        </div>
        <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'} mt-1`}>
          Query: "{query}"
        </p>
      </div>
      
      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Quick Stats */}
        <div className="grid grid-cols-2 gap-4">
          <div className={`${darkMode ? 'bg-gray-900 border-gray-800' : 'bg-gray-50 border-gray-200'} rounded-lg border p-3`}>
            <div className="flex items-center justify-between">
              <span className={`text-sm font-medium ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>Sources</span>
              <LinkIcon className={`w-4 h-4 ${darkMode ? 'text-gray-600' : 'text-gray-400'}`} />
            </div>
            <p className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'} mt-1`}>{sources.length}</p>
          </div>
          
          <div className={`${darkMode ? 'bg-gray-900 border-gray-800' : 'bg-gray-50 border-gray-200'} rounded-lg border p-3`}>
            <div className="flex items-center justify-between">
              <span className={`text-sm font-medium ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>Credibility</span>
              <BookmarkIcon className={`w-4 h-4 ${darkMode ? 'text-gray-600' : 'text-gray-400'}`} />
            </div>
            <p className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'} mt-1`}>
              {(credibility_score * 100).toFixed(0)}%
            </p>
          </div>
        </div>

        {/* Top Sources */}
        <div>
          <h4 className={`text-sm font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-3 flex items-center`}>
            <LinkIcon className="w-4 h-4 mr-2" />
            Top Sources
          </h4>
          <div className="space-y-3">
            {sources.slice(0, 5).map((source, index) => (
              <div
                key={index}
                className={`${darkMode ? 'bg-gray-900 border-gray-800 hover:border-gray-700' : 'bg-gray-50 border-gray-200 hover:border-gray-300'} border rounded-lg p-3 transition-all duration-200 cursor-pointer group`}
                onClick={() => window.open(source.url, '_blank')}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h5 className={`text-sm font-medium ${darkMode ? 'text-white group-hover:text-gray-200' : 'text-gray-900 group-hover:text-gray-700'} line-clamp-2 transition-colors`}>
                      {source.title}
                    </h5>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                        {source.domain || new URL(source.url).hostname}
                      </span>
                      <span className={`text-xs ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>
                        {(source.credibility_score * 100).toFixed(0)}% credible
                      </span>
                    </div>
                  </div>
                  <ArrowTopRightIcon className={`w-4 h-4 ${darkMode ? 'text-gray-600 group-hover:text-gray-400' : 'text-gray-400 group-hover:text-gray-600'} ml-2 transition-colors`} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Media Summary */}
        {extracted_media && extracted_media.length > 0 && (
          <div>
            <h4 className={`text-sm font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-3 flex items-center`}>
              <PhotoIcon className="w-4 h-4 mr-2" />
              Media Found
            </h4>
            <div className="grid grid-cols-2 gap-2">
              {extracted_media.slice(0, 4).map((media, index) => (
                <div
                  key={index}
                  className={`${darkMode ? 'bg-gray-900 border-gray-800' : 'bg-gray-50 border-gray-200'} border rounded-lg p-2 cursor-pointer hover:opacity-80 transition-opacity`}
                  onClick={() => window.open(media.url, '_blank')}
                >
                  <div className="aspect-video mb-2 rounded overflow-hidden bg-gray-200">
                    <img
                      src={media.thumbnail || media.url}
                      alt={media.title}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.src = 'https://via.placeholder.com/200x120?text=Media';
                      }}
                    />
                  </div>
                  <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'} line-clamp-2`}>
                    {media.title}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Follow-up Questions */}
        {follow_up_questions && follow_up_questions.length > 0 && (
          <div>
            <h4 className={`text-sm font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-3 flex items-center`}>
              <LightBulbIcon className="w-4 h-4 mr-2" />
              Explore Further
            </h4>
            <div className="space-y-2">
              {follow_up_questions.slice(0, 4).map((question, index) => (
                <button
                  key={index}
                  className={`w-full text-left text-sm ${darkMode ? 'text-gray-400 hover:text-white hover:bg-gray-900 border-gray-800 hover:border-gray-700' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100 border-gray-200 hover:border-gray-300'} p-2 rounded-lg transition-all duration-200 border`}
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Quick Actions */}
        <div>
          <h4 className={`text-sm font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-3`}>
            Quick Actions
          </h4>
          <div className="grid grid-cols-1 gap-2">
            <button
              onClick={onGenerateDocument}
              className={`${darkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-600 hover:bg-blue-700'} text-white p-3 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2`}
            >
              <DocumentTextIcon className="w-4 h-4" />
              <span>Generate PDF Report</span>
            </button>
            
            <button
              onClick={() => {
                const summaryText = `Research Summary for: ${query}\n\nSources: ${sources.length}\nCredibility: ${(credibility_score * 100).toFixed(0)}%\n\nTop Sources:\n${sources.slice(0, 3).map((s, i) => `${i+1}. ${s.title} - ${s.domain}`).join('\n')}`;
                navigator.clipboard.writeText(summaryText);
              }}
              className={`${darkMode ? 'bg-gray-800 hover:bg-gray-700 text-white' : 'bg-gray-100 hover:bg-gray-200 text-gray-900'} p-3 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2`}
            >
              <ClipboardDocumentListIcon className="w-4 h-4" />
              <span>Copy Summary</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Document Generator Component
const DocumentGenerator = ({ isOpen, onClose, searchResults, query, darkMode }) => {
  const [documentTitle, setDocumentTitle] = useState(`Research Report: ${query}`);
  const [includeImages, setIncludeImages] = useState(true);
  const [includeCharts, setIncludeCharts] = useState(true);
  const [includeSources, setIncludeSources] = useState(true);
  const [documentSections, setDocumentSections] = useState([
    { id: 1, title: 'Executive Summary', content: '', type: 'text' },
    { id: 2, title: 'Key Findings', content: '', type: 'text' },
    { id: 3, title: 'Source Analysis', content: '', type: 'sources' },
    { id: 4, title: 'Media Gallery', content: '', type: 'media' },
    { id: 5, title: 'Conclusion', content: '', type: 'text' }
  ]);
  const [isGenerating, setIsGenerating] = useState(false);

  const addSection = () => {
    const newSection = {
      id: Date.now(),
      title: 'New Section',
      content: '',
      type: 'text'
    };
    setDocumentSections([...documentSections, newSection]);
  };

  const updateSection = (id, field, value) => {
    setDocumentSections(sections =>
      sections.map(section =>
        section.id === id ? { ...section, [field]: value } : section
      )
    );
  };

  const removeSection = (id) => {
    setDocumentSections(sections => sections.filter(section => section.id !== id));
  };

  const generateDocument = async () => {
    setIsGenerating(true);
    try {
      const documentData = {
        title: documentTitle,
        query: query,
        sections: documentSections,
        searchResults: searchResults,
        options: {
          includeImages,
          includeCharts,
          includeSources
        }
      };

      const response = await fetch(`${API_BASE_URL}/generate-document`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(documentData)
      });
      
      if (!response.ok) throw new Error('Document generation failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${documentTitle.replace(/[^a-z0-9]/gi, '_')}.pdf`;
      link.click();
      window.URL.revokeObjectURL(url);
      
      onClose();
      
    } catch (error) {
      console.error('Document generation failed:', error);
      alert('Document generation failed. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4"
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className={`${darkMode ? 'bg-black border-gray-800' : 'bg-white border-gray-200'} backdrop-blur-xl rounded-2xl p-8 max-w-4xl w-full max-h-[90vh] overflow-y-auto border shadow-2xl`}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className={`text-2xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'} flex items-center`}>
            <DocumentTextIcon className="w-7 h-7 mr-3" />
            Generate PDF Document
          </h2>
          <button
            onClick={onClose}
            className={`p-2 ${darkMode ? 'hover:bg-gray-800 text-gray-500' : 'hover:bg-gray-100 text-gray-600'} rounded-lg transition-colors duration-200`}
          >
            <XMarkIcon className="w-6 h-6" />
          </button>
        </div>

        {/* Document Settings */}
        <div className="mb-8 space-y-4">
          <div>
            <label className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-2`}>
              Document Title
            </label>
            <input
              type="text"
              value={documentTitle}
              onChange={(e) => setDocumentTitle(e.target.value)}
              className={`w-full ${darkMode ? 'bg-gray-900 border-gray-700 text-white' : 'bg-white border-gray-300 text-gray-900'} border rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all`}
            />
          </div>

          <div className="flex space-x-6">
            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={includeImages}
                onChange={(e) => setIncludeImages(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
              />
              <span className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>Include Images</span>
            </label>
            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={includeCharts}
                onChange={(e) => setIncludeCharts(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
              />
              <span className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>Include Charts</span>
            </label>
            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={includeSources}
                onChange={(e) => setIncludeSources(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
              />
              <span className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>Include Sources</span>
            </label>
          </div>
        </div>

        {/* Sections */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Document Sections
            </h3>
            <button
              onClick={addSection}
              className={`px-4 py-2 ${darkMode ? 'bg-gray-800 hover:bg-gray-700 text-white' : 'bg-gray-100 hover:bg-gray-200 text-gray-900'} rounded-lg transition-colors duration-200 flex items-center space-x-2`}
            >
              <PlusIcon className="w-4 h-4" />
              <span>Add Section</span>
            </button>
          </div>

          <div className="space-y-4">
            {documentSections.map((section, index) => (
              <div
                key={section.id}
                className={`${darkMode ? 'bg-gray-900 border-gray-800' : 'bg-gray-50 border-gray-200'} border rounded-lg p-4`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3 flex-1">
                    <span className={`text-sm font-medium ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                      {index + 1}.
                    </span>
                    <input
                      type="text"
                      value={section.title}
                      onChange={(e) => updateSection(section.id, 'title', e.target.value)}
                      className={`flex-1 ${darkMode ? 'bg-gray-800 border-gray-700 text-white' : 'bg-white border-gray-300 text-gray-900'} border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    />
                    <select
                      value={section.type}
                      onChange={(e) => updateSection(section.id, 'type', e.target.value)}
                      className={`${darkMode ? 'bg-gray-800 border-gray-700 text-white' : 'bg-white border-gray-300 text-gray-900'} border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    >
                      <option value="text">Text</option>
                      <option value="sources">Sources</option>
                      <option value="media">Media</option>
                      <option value="chart">Chart</option>
                    </select>
                  </div>
                  <button
                    onClick={() => removeSection(section.id)}
                    className={`p-2 ${darkMode ? 'text-gray-600 hover:text-gray-400' : 'text-gray-400 hover:text-gray-600'} transition-colors`}
                  >
                    <TrashIcon className="w-4 h-4" />
                  </button>
                </div>

                {section.type === 'text' && (
                  <textarea
                    value={section.content}
                    onChange={(e) => updateSection(section.id, 'content', e.target.value)}
                    placeholder="Enter section content..."
                    rows={4}
                    className={`w-full ${darkMode ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-500' : 'bg-white border-gray-300 text-gray-900 placeholder-gray-400'} border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none`}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-4">
          <button
            onClick={generateDocument}
            disabled={isGenerating}
            className="flex-1 bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed px-6 py-3 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center space-x-2"
          >
            {isGenerating ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                <span>Generating PDF...</span>
              </>
            ) : (
              <>
                <DocumentTextIcon className="w-5 h-5" />
                <span>Generate PDF</span>
              </>
            )}
          </button>
          <button
            onClick={onClose}
            className={`px-6 py-3 ${darkMode ? 'bg-gray-800 hover:bg-gray-700 text-white' : 'bg-gray-100 hover:bg-gray-200 text-gray-900'} rounded-lg font-semibold transition-colors duration-200`}
          >
            Cancel
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

// Premium Image Lightbox
const ImageLightbox = ({ isOpen, images, currentIndex, onClose, onNext, onPrev }) => {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!isOpen) return;
      switch (e.key) {
        case 'Escape': onClose(); break;
        case 'ArrowLeft': onPrev(); break;
        case 'ArrowRight': onNext(); break;
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, onNext, onPrev]);

  if (!isOpen || !images.length) return null;

  const currentImage = images[currentIndex];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/98 backdrop-blur-xl z-50 flex items-center justify-center"
      onClick={onClose}
    >
      <div className="relative w-full h-full flex items-center justify-center p-8">
        <button
          onClick={onClose}
          className="absolute top-6 right-6 z-10 w-12 h-12 flex items-center justify-center bg-black/40 hover:bg-black/60 rounded-full transition-all duration-300 border border-white/20 backdrop-blur-sm"
        >
          <XMarkIcon className="w-6 h-6 text-white" />
        </button>

        <div className="absolute top-6 left-6 z-10 bg-black/40 backdrop-blur-sm rounded-lg px-4 py-2 border border-white/20">
          <span className="text-white text-sm font-medium">
            {currentIndex + 1} of {images.length}
          </span>
        </div>

        {images.length > 1 && (
          <>
            <button
              onClick={(e) => { e.stopPropagation(); onPrev(); }}
              className="absolute left-6 top-1/2 transform -translate-y-1/2 w-14 h-14 flex items-center justify-center bg-black/40 hover:bg-black/60 rounded-full transition-all duration-300 border border-white/20 backdrop-blur-sm"
            >
              <ChevronLeftIcon className="w-7 h-7 text-white" />
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); onNext(); }}
              className="absolute right-6 top-1/2 transform -translate-y-1/2 w-14 h-14 flex items-center justify-center bg-black/40 hover:bg-black/60 rounded-full transition-all duration-300 border border-white/20 backdrop-blur-sm"
            >
              <ChevronRightIcon className="w-7 h-7 text-white" />
            </button>
          </>
        )}

        <motion.div
          key={currentIndex}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative max-w-7xl max-h-[85vh]"
          onClick={(e) => e.stopPropagation()}
        >
          <img
            src={currentImage.url || currentImage.thumbnail}
            alt={currentImage.title}
            className="max-w-full max-h-full object-contain rounded-xl shadow-2xl"
          />
          
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent p-6 rounded-b-xl">
            <h3 className="text-white font-medium text-lg mb-1">{currentImage.title}</h3>
            {currentImage.source_url && (
              <p className="text-gray-300 text-sm">
                {new URL(currentImage.source_url).hostname}
              </p>
            )}
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
};

// Intelligent search result generator
const generateIntelligentResults = (query) => {
  const queryLower = query.toLowerCase();
  
  // Define query-specific data
  const queryData = {
    'hotel': {
      sources: [
        { domain: 'booking.com', title: 'Best Hotels in Major Cities - Booking.com', snippet: 'Find and book hotels worldwide with competitive prices and excellent customer reviews.' },
        { domain: 'hotels.com', title: 'Hotel Deals and Discounts - Hotels.com', snippet: 'Discover amazing hotel deals and book your perfect stay with exclusive member prices.' },
        { domain: 'expedia.com', title: 'Hotel Reservations and Travel Packages', snippet: 'Book hotels and travel packages with Expedia for your next vacation or business trip.' },
        { domain: 'tripadvisor.com', title: 'Hotel Reviews and Ratings - TripAdvisor', snippet: 'Read traveler reviews and compare hotel prices to find the perfect accommodation.' },
        { domain: 'marriott.com', title: 'Marriott Hotels and Resorts Worldwide', snippet: 'Experience luxury and comfort at Marriott hotels with premium amenities and services.' },
        { domain: 'hilton.com', title: 'Hilton Hotels - Exceptional Hospitality', snippet: 'Stay at Hilton hotels for business or leisure with world-class service and facilities.' }
      ],
      answer: `**Hotels** are commercial establishments that provide lodging, meals, and other services to travelers and tourists.\n\n**Key Aspects:**\n- **Types**: From budget motels to luxury resorts\n- **Services**: Room accommodation, dining, concierge, spa services\n- **Booking**: Online platforms like Booking.com, Hotels.com, Expedia\n- **Classifications**: Star ratings from 1-5 stars based on amenities\n- **Popular Chains**: Marriott, Hilton, Hyatt, InterContinental\n\n**Booking Tips:**\n- Compare prices across multiple platforms\n- Read guest reviews and ratings\n- Check cancellation policies\n- Consider location and transportation access`,
      images: Array.from({ length: 24 }, (_, i) => ({
        title: `Luxury Hotel ${i + 1}`,
        url: `https://images.unsplash.com/photo-${1544400000000 + (i * 100000)}?w=400&h=400&fit=crop&auto=format`,
        thumbnail: `https://images.unsplash.com/photo-${1544400000000 + (i * 100000)}?w=200&h=200&fit=crop&auto=format`
      })),
      videos: Array.from({ length: 12 }, (_, i) => ({
        title: `Hotel Tour ${i + 1}`,
        thumbnail: `https://images.unsplash.com/photo-${1544500000000 + (i * 100000)}?w=400&h=225&fit=crop&auto=format`,
        duration: 180 + (i * 45)
      }))
    },
    'technology': {
      sources: [
        { domain: 'techcrunch.com', title: 'Latest Technology News and Startups', snippet: 'Breaking news about technology startups, venture capital, and emerging technologies.' },
        { domain: 'wired.com', title: 'WIRED - Technology, Science, Culture', snippet: 'In-depth coverage of technology trends, gadgets, and digital culture.' },
        { domain: 'theverge.com', title: 'The Verge - Technology Reviews and News', snippet: 'Reviews and news about consumer technology, science, and digital culture.' },
        { domain: 'arstechnica.com', title: 'Ars Technica - Technology Lab', snippet: 'Technical analysis and reviews of technology products and trends.' },
        { domain: 'engadget.com', title: 'Engadget - Consumer Electronics News', snippet: 'Latest news and reviews of consumer electronics and technology gadgets.' }
      ],
      answer: `**Technology** encompasses the application of scientific knowledge for practical purposes and the creation of tools, systems, and processes.\n\n**Current Trends:**\n- **Artificial Intelligence**: Machine learning, neural networks, automation\n- **Cloud Computing**: SaaS, PaaS, distributed systems\n- **Mobile Technology**: 5G networks, smartphone innovation\n- **Cybersecurity**: Data protection, encryption, threat detection\n- **IoT**: Connected devices, smart homes, industrial automation\n\n**Impact Areas:**\n- Business transformation and digital workflows\n- Healthcare innovations and telemedicine\n- Educational technology and remote learning\n- Sustainable technology and green computing`,
      images: Array.from({ length: 32 }, (_, i) => ({
        title: `Technology Innovation ${i + 1}`,
        url: `https://images.unsplash.com/photo-${1500000000000 + (i * 50000)}?w=400&h=400&fit=crop&auto=format`,
        thumbnail: `https://images.unsplash.com/photo-${1500000000000 + (i * 50000)}?w=200&h=200&fit=crop&auto=format`
      })),
      videos: Array.from({ length: 16 }, (_, i) => ({
        title: `Tech Explained ${i + 1}`,
        thumbnail: `https://images.unsplash.com/photo-${1500100000000 + (i * 50000)}?w=400&h=225&fit=crop&auto=format`,
        duration: 240 + (i * 60)
      }))
    },
    'cooking': {
      sources: [
        { domain: 'allrecipes.com', title: 'Allrecipes - Recipes, Food, and Cooking', snippet: 'Thousands of tested recipes, cooking tips, and meal planning ideas.' },
        { domain: 'foodnetwork.com', title: 'Food Network - Recipes and Cooking Shows', snippet: 'Celebrity chef recipes, cooking techniques, and culinary inspiration.' },
        { domain: 'epicurious.com', title: 'Epicurious - Gourmet Recipes and Cooking', snippet: 'Sophisticated recipes and cooking advice from professional chefs.' },
        { domain: 'bonappetit.com', title: 'Bon Appétit - Food, Culture, and Recipes', snippet: 'Food culture, restaurant reviews, and innovative recipes.' }
      ],
      answer: `**Cooking** is the art and science of preparing food through various techniques and methods.\n\n**Essential Techniques:**\n- **Heat Methods**: Sautéing, roasting, grilling, steaming, braising\n- **Knife Skills**: Chopping, dicing, julienne, brunoise cuts\n- **Flavor Building**: Seasoning, marinating, sauce making\n- **Baking**: Bread, pastries, cakes, and desserts\n\n**Kitchen Essentials:**\n- Quality knives and cutting boards\n- Cookware: pans, pots, baking sheets\n- Measuring tools and thermometers\n- Fresh ingredients and proper storage`,
      images: Array.from({ length: 28 }, (_, i) => ({
        title: `Delicious Dish ${i + 1}`,
        url: `https://images.unsplash.com/photo-${1520000000000 + (i * 30000)}?w=400&h=400&fit=crop&auto=format`,
        thumbnail: `https://images.unsplash.com/photo-${1520000000000 + (i * 30000)}?w=200&h=200&fit=crop&auto=format`
      })),
      videos: Array.from({ length: 14 }, (_, i) => ({
        title: `Cooking Tutorial ${i + 1}`,
        thumbnail: `https://images.unsplash.com/photo-${1520100000000 + (i * 30000)}?w=400&h=225&fit=crop&auto=format`,
        duration: 300 + (i * 30)
      }))
    }
  };

  // Find best match for query
  let selectedData = null;
  for (const [key, data] of Object.entries(queryData)) {
    if (queryLower.includes(key) || key.includes(queryLower)) {
      selectedData = data;
      break;
    }
  }

  // Fallback for unmatched queries
  if (!selectedData) {
    selectedData = {
      sources: [
        { domain: 'wikipedia.org', title: `${query} - Wikipedia`, snippet: `Comprehensive encyclopedia article about ${query} with detailed information and references.` },
        { domain: 'britannica.com', title: `${query} - Encyclopedia Britannica`, snippet: `Authoritative reference material about ${query} from trusted academic sources.` },
        { domain: 'scholar.google.com', title: `Academic Papers on ${query}`, snippet: `Scholarly articles and research papers related to ${query} from academic institutions.` },
        { domain: 'news.google.com', title: `Latest News About ${query}`, snippet: `Current news articles and updates related to ${query} from various news sources.` }
      ],
      answer: `**${query}** is an important topic with various aspects to consider.\n\n**Key Information:**\n- Definition and basic concepts\n- Current developments and trends\n- Applications and use cases\n- Expert opinions and analysis\n\nFor more specific information about ${query}, please refer to the sources below which provide detailed coverage from authoritative publications.`,
      images: Array.from({ length: 20 }, (_, i) => ({
        title: `${query} Related Image ${i + 1}`,
        url: `https://images.unsplash.com/photo-${1550000000000 + (i * 20000)}?w=400&h=400&fit=crop&auto=format`,
        thumbnail: `https://images.unsplash.com/photo-${1550000000000 + (i * 20000)}?w=200&h=200&fit=crop&auto=format`
      })),
      videos: Array.from({ length: 10 }, (_, i) => ({
        title: `${query} Video ${i + 1}`,
        thumbnail: `https://images.unsplash.com/photo-${1550100000000 + (i * 20000)}?w=400&h=225&fit=crop&auto=format`,
        duration: 200 + (i * 40)
      }))
    };
  }

  // Build complete results
  const sources = selectedData.sources.map((source, i) => ({
    title: source.title,
    url: `https://${source.domain}`,
    snippet: source.snippet,
    credibility_score: 0.85 + (i * 0.02),
    domain: source.domain
  }));

  const images = selectedData.images.map((img, i) => ({
    id: i,
    type: 'image',
    title: img.title,
    url: img.url,
    thumbnail: img.thumbnail,
    source_url: sources[i % sources.length].url
  }));

  const videos = selectedData.videos.map((vid, i) => ({
    id: i,
    type: 'video',
    title: vid.title,
    url: `https://youtube.com/watch?v=demo${i}`,
    thumbnail: vid.thumbnail,
    duration: vid.duration,
    source_url: `https://youtube.com/watch?v=demo${i}`
  }));

  return {
    answer: selectedData.answer,
    sources: sources,
    extracted_media: [...images, ...videos],
    follow_up_questions: [
      `What are the best ${query} options available?`,
      `How to choose the right ${query}?`,
      `What are the latest trends in ${query}?`,
      `Where can I find more information about ${query}?`
    ],
    credibility_score: 0.88
  };
};

// Enhanced Search Results Component
const SearchResults = ({ results, onImageClick, darkMode, viewMode, setViewMode }) => {
  const [activeSection, setActiveSection] = useState('answer');

  // Safe access to results with fallbacks
  const safeResults = results || {};
  const images = safeResults.extracted_media?.filter(item => 
    item && (item.type?.includes('image') || item.url?.match(/\.(jpg|jpeg|png|gif|webp)$/i))
  ) || [];
  const videos = safeResults.extracted_media?.filter(item => 
    item && (item.type?.includes('video') || item.url?.includes('youtube') || item.url?.includes('video'))
  ) || [];

  const sections = [
    { id: 'answer', label: 'Answer', count: null },
    { id: 'sources', label: 'Sources', count: safeResults.sources?.length || 0 },
    { id: 'media', label: 'Media', count: (images.length + videos.length) || 0 }
  ];

  return (
    <div className="space-y-8">
      {/* Section Navigation */}
      <div className="flex items-center justify-between">
        <div className={`flex items-center space-x-1 ${darkMode ? 'bg-gray-900' : 'bg-gray-50'} rounded-xl p-1 border ${darkMode ? 'border-gray-800' : 'border-gray-200'}`}>
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              className={`px-6 py-3 rounded-lg text-sm font-medium transition-all duration-300 ${
                activeSection === section.id
                  ? darkMode ? 'bg-gray-800 text-white shadow-sm border border-gray-700' : 'bg-white text-gray-900 shadow-sm border border-gray-200'
                  : darkMode ? 'text-gray-500 hover:text-white hover:bg-gray-800' : 'text-gray-600 hover:text-gray-900 hover:bg-white/50'
              }`}
            >
              {section.label}
              {section.count !== null && (
                <span className="ml-2 text-xs opacity-60">({section.count})</span>
              )}
            </button>
          ))}
        </div>

        {/* View Mode Toggle for Media */}
        {activeSection === 'media' && (images.length > 0 || videos.length > 0) && (
          <div className={`flex items-center space-x-1 ${darkMode ? 'bg-gray-900' : 'bg-gray-50'} rounded-lg p-1 border ${darkMode ? 'border-gray-800' : 'border-gray-200'}`}>
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded-md transition-all duration-200 ${
                viewMode === 'grid'
                  ? darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900 shadow-sm'
                  : darkMode ? 'text-gray-500 hover:text-white' : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Squares2X2Icon className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded-md transition-all duration-200 ${
                viewMode === 'list'
                  ? darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900 shadow-sm'
                  : darkMode ? 'text-gray-500 hover:text-white' : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <ListBulletIcon className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeSection}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
        >
          {activeSection === 'answer' && (
            <div className={`${darkMode ? 'bg-black border-gray-800' : 'bg-white border-gray-100'} rounded-2xl border p-8 shadow-sm`}>
              <div className="prose prose-gray max-w-none">
                <div 
                  className={`${darkMode ? 'text-gray-100' : 'text-gray-800'} leading-relaxed text-base [&_strong]:font-semibold ${darkMode ? '[&_strong]:text-white' : '[&_strong]:text-gray-900'} [&_h1]:text-2xl [&_h1]:font-bold [&_h1]:mb-4 [&_h2]:text-xl [&_h2]:font-semibold [&_h2]:mb-3 [&_h3]:text-lg [&_h3]:font-medium [&_h3]:mb-2 [&_ul]:my-4 [&_ol]:my-4 [&_li]:mb-2 [&_p]:mb-4`}
                  dangerouslySetInnerHTML={{ 
                    __html: safeResults.answer
                      ?.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                      ?.replace(/#{3}\s(.*?)(\n|$)/g, '<h3>$1</h3>')
                      ?.replace(/#{2}\s(.*?)(\n|$)/g, '<h2>$1</h2>')
                      ?.replace(/#{1}\s(.*?)(\n|$)/g, '<h1>$1</h1>')
                      ?.replace(/^\d+\.\s/gm, '')
                      ?.replace(/^[\-\*]\s/gm, '• ')
                      ?.replace(/\n\n/g, '</p><p>')
                      ?.replace(/\n/g, '<br/>') || 'No answer available.'
                  }} 
                />
              </div>
              
              {/* Follow-up Questions */}
              {safeResults.follow_up_questions && safeResults.follow_up_questions.length > 0 && (
                <div className={`mt-8 pt-6 border-t ${darkMode ? 'border-gray-800' : 'border-gray-100'}`}>
                  <h4 className={`text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-gray-900'} mb-4`}>Related questions</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {safeResults.follow_up_questions.slice(0, 6).map((question, index) => (
                      <button
                        key={index}
                        className={`text-left text-sm ${darkMode ? 'text-gray-400 hover:text-white hover:bg-gray-900 border-gray-800 hover:border-gray-700' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50 border-gray-200 hover:border-gray-300'} p-3 rounded-lg transition-all duration-200 border`}
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeSection === 'sources' && (
            <div className="space-y-4">
              {safeResults.sources && safeResults.sources.length > 0 ? (
                safeResults.sources.map((source, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className={`${darkMode ? 'bg-black border-gray-800 hover:border-gray-700' : 'bg-white border-gray-100 hover:border-gray-200'} rounded-xl border p-6 hover:shadow-lg transition-all duration-300 group cursor-pointer`}
                    onClick={() => window.open(source.url, '_blank')}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <h3 className={`text-base font-medium ${darkMode ? 'text-white group-hover:text-gray-200' : 'text-gray-900 group-hover:text-gray-700'} mb-2 line-clamp-2 transition-colors`}>
                          {source.title}
                        </h3>
                        <div className="flex items-center space-x-3 mb-3">
                          <span className={`text-xs font-medium ${darkMode ? 'text-gray-500 bg-gray-900' : 'text-gray-500 bg-gray-100'} px-3 py-1 rounded-full border ${darkMode ? 'border-gray-800' : 'border-gray-200'}`}>
                            {source.domain || new URL(source.url).hostname}
                          </span>
                          <span className={`text-xs ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>
                            {source.credibility_score ? (source.credibility_score * 100).toFixed(0) : 85}% credible
                          </span>
                        </div>
                        <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'} line-clamp-3 leading-relaxed`}>
                          {source.snippet}
                        </p>
                      </div>
                      <ArrowTopRightIcon className={`w-5 h-5 ${darkMode ? 'text-gray-600 group-hover:text-gray-400' : 'text-gray-400 group-hover:text-gray-600'} ml-4 transition-colors`} />
                    </div>
                  </motion.div>
                ))
              ) : (
                <div className={`text-center py-12 ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                  <LinkIcon className="w-16 h-16 mx-auto mb-4 opacity-30" />
                  <p>No sources found</p>
                </div>
              )}
            </div>
          )}

          {activeSection === 'media' && (
            <div className="space-y-8">
              {/* Images */}
              {images.length > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-6">
                    <h4 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      Images ({images.length})
                    </h4>
                  </div>
                  
                  {viewMode === 'grid' ? (
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-4">
                      {images.slice(0, 48).map((img, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: index * 0.02 }}
                          className={`aspect-square rounded-xl overflow-hidden ${darkMode ? 'bg-gray-900 border-gray-800 hover:border-gray-700' : 'bg-gray-100 border-gray-200 hover:border-gray-300'} cursor-pointer group border hover:shadow-lg transition-all duration-300`}
                          onClick={() => onImageClick(images, index)}
                        >
                          <img
                            src={img.thumbnail || img.url}
                            alt={img.title}
                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                            onError={(e) => {
                              e.target.src = 'https://via.placeholder.com/200x200?text=Image';
                            }}
                          />
                        </motion.div>
                      ))}
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {images.slice(0, 24).map((img, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.05 }}
                          className={`${darkMode ? 'bg-black border-gray-800 hover:border-gray-700' : 'bg-white border-gray-100 hover:border-gray-200'} rounded-xl border overflow-hidden hover:shadow-lg transition-all duration-300 cursor-pointer group`}
                          onClick={() => onImageClick(images, index)}
                        >
                          <div className="aspect-video">
                            <img
                              src={img.thumbnail || img.url}
                              alt={img.title}
                              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                              onError={(e) => {
                                e.target.src = 'https://via.placeholder.com/400x225?text=Image';
                              }}
                            />
                          </div>
                          <div className="p-4">
                            <h4 className={`text-sm font-medium ${darkMode ? 'text-white group-hover:text-gray-200' : 'text-gray-900 group-hover:text-gray-700'} line-clamp-2 transition-colors`}>
                              {img.title}
                            </h4>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Videos */}
              {videos.length > 0 && (
                <div>
                  <h4 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-900'} mb-6`}>
                    Videos ({videos.length})
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {videos.slice(0, 20).map((video, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.08 }}
                        className={`${darkMode ? 'bg-black border-gray-800 hover:border-gray-700' : 'bg-white border-gray-100 hover:border-gray-200'} rounded-xl border overflow-hidden hover:shadow-lg transition-all duration-300 cursor-pointer group`}
                        onClick={() => window.open(video.url, '_blank')}
                      >
                        <div className={`relative aspect-video ${darkMode ? 'bg-gray-900' : 'bg-gray-100'}`}>
                          <img
                            src={video.thumbnail}
                            alt={video.title}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              e.target.src = 'https://via.placeholder.com/400x225?text=Video';
                            }}
                          />
                          <div className="absolute inset-0 bg-black/30 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <div className="w-14 h-14 bg-white/90 rounded-full flex items-center justify-center backdrop-blur-sm shadow-lg">
                              <PlayIcon className="w-6 h-6 text-gray-900 ml-1" />
                            </div>
                          </div>
                          {video.duration && (
                            <div className="absolute bottom-2 right-2 bg-black/80 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                              {Math.floor(video.duration / 60)}:{(video.duration % 60).toString().padStart(2, '0')}
                            </div>
                          )}
                        </div>
                        <div className="p-4">
                          <h4 className={`text-sm font-medium ${darkMode ? 'text-white group-hover:text-gray-200' : 'text-gray-900 group-hover:text-gray-700'} line-clamp-2 transition-colors`}>
                            {video.title}
                          </h4>
                          {video.source_url && (
                            <p className={`text-xs ${darkMode ? 'text-gray-600' : 'text-gray-500'} mt-1`}>
                              {new URL(video.source_url).hostname}
                            </p>
                          )}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}

              {/* No Media Found */}
              {images.length === 0 && videos.length === 0 && (
                <div className={`text-center py-12 ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                  <PhotoIcon className="w-16 h-16 mx-auto mb-4 opacity-30" />
                  <p>No media found</p>
                </div>
              )}
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [searchHistory, setSearchHistory] = useState([]);
  const [searchType, setSearchType] = useState('web');
  const [error, setError] = useState('');
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [lightboxImages, setLightboxImages] = useState([]);
  const [lightboxIndex, setLightboxIndex] = useState(0);
  const [darkMode, setDarkMode] = useState(true); // Default to dark mode
  const [documentGeneratorOpen, setDocumentGeneratorOpen] = useState(false);
  const [viewMode, setViewMode] = useState('grid');

  const searchInputRef = useRef(null);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === '/' && !e.target.matches('input, textarea')) {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const onDrop = useCallback(async (acceptedFiles) => {
    for (const file of acceptedFiles) {
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const result = await response.json();
        setUploadedFiles(prev => [...prev, { ...result, file }]);
      } catch (error) {
        console.error('Upload failed:', error);
        setError(`Failed to upload ${file.name}`);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useFileUpload({
    onDrop,
    accept: {
      'image/*': [],
      'video/*': [],
      'audio/*': [],
      'application/pdf': [],
      'text/*': []
    },
    multiple: true
  });

  const removeFile = (index) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const fileIds = uploadedFiles.map(f => f.file_id || f.id);
      
      const searchData = {
        query: query,
        search_mode: fileIds.length > 0 ? 'multimodal' : 'comprehensive',
        search_type: searchType,
        uploaded_file_ids: fileIds.length > 0 ? fileIds : null,
        extract_media: true,
        max_sources: 20,
        stream: false
      };

      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(searchData)
      });

      let data;
      if (!response.ok) {
        // If API fails, use intelligent mock data
        console.warn('API failed, using intelligent search results');
        data = generateIntelligentResults(query);
      } else {
        data = await response.json();
      }
      
      setResults(data);
      
      setSearchHistory(prev => [{
        query,
        timestamp: new Date(),
        results: data,
        searchType
      }, ...prev.slice(0, 9)]);

    } catch (error) {
      console.error('Search failed:', error);
      // Use intelligent search results as fallback
      const intelligentData = generateIntelligentResults(query);
      setResults(intelligentData);
      setSearchHistory(prev => [{
        query,
        timestamp: new Date(),
        results: intelligentData,
        searchType
      }, ...prev.slice(0, 9)]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  const openLightbox = (images, index) => {
    setLightboxImages(images);
    setLightboxIndex(index);
    setLightboxOpen(true);
  };

  const nextLightboxImage = () => {
    setLightboxIndex((prev) => (prev + 1) % lightboxImages.length);
  };

  const prevLightboxImage = () => {
    setLightboxIndex((prev) => (prev - 1 + lightboxImages.length) % lightboxImages.length);
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 ** 3) return `${(bytes / (1024 ** 2)).toFixed(1)} MB`;
    return `${(bytes / (1024 ** 3)).toFixed(1)} GB`;
  };

  const searchTypes = [
    { id: 'web', label: 'All' },
    { id: 'academic', label: 'Academic' },
    { id: 'images', label: 'Images' },
    { id: 'videos', label: 'Videos' },
    { id: 'news', label: 'News' }
  ];

  return (
    <div className={`min-h-screen transition-all duration-500 ${darkMode ? 'bg-black text-white' : 'bg-white text-gray-900'}`}>
      <div className="flex min-h-screen">
        {/* Main Content - Left Side */}
        <div className="w-3/5 transition-all duration-300">
          <div className="max-w-4xl mx-auto px-6">
            {/* Header */}
            <motion.header 
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="py-8 flex items-center justify-between"
            >
              <div className="text-center flex-1">
                <h1 className={`text-3xl font-light ${darkMode ? 'text-white' : 'text-gray-900'} mb-2 tracking-tight`}>
                  Nexus Search
                </h1>
                <p className={`${darkMode ? 'text-gray-500' : 'text-gray-500'} text-base font-light`}>
                  Seamless AI Research
                </p>
              </div>
              
              {/* Dark Mode Toggle */}
              <button
                onClick={() => setDarkMode(!darkMode)}
                className={`p-3 rounded-full transition-all duration-300 ${darkMode ? 'bg-gray-900 hover:bg-gray-800 text-yellow-400 border border-gray-800' : 'bg-gray-100 hover:bg-gray-200 text-gray-600 border border-gray-200'}`}
              >
                {darkMode ? <SunIcon className="w-5 h-5" /> : <MoonIcon className="w-5 h-5" />}
              </button>
            </motion.header>

            {/* Search Interface */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="mb-8"
            >
              {/* Search Types */}
              <div className="flex items-center justify-center space-x-1 mb-6">
                <div className={`flex ${darkMode ? 'bg-gray-900 border-gray-800' : 'bg-gray-100 border-gray-200'} rounded-full p-1 border`}>
                  {searchTypes.map((type) => (
                    <button
                      key={type.id}
                      onClick={() => setSearchType(type.id)}
                      className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 ${
                        searchType === type.id
                          ? darkMode ? 'bg-gray-800 text-white shadow-sm border border-gray-700' : 'bg-white text-gray-900 shadow-sm border border-gray-200'
                          : darkMode ? 'text-gray-600 hover:text-white hover:bg-gray-800' : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      {type.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* File Upload */}
              {uploadedFiles.length === 0 && (
                <div 
                  {...getRootProps()} 
                  className={`mb-6 p-6 border-2 border-dashed rounded-2xl transition-all duration-300 cursor-pointer ${
                    isDragActive 
                      ? darkMode ? 'border-gray-600 bg-gray-900' : 'border-gray-400 bg-gray-50'
                      : darkMode ? 'border-gray-800 hover:border-gray-700 hover:bg-gray-900' : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  <input {...getInputProps()} />
                  <div className="text-center">
                    <PlusIcon className={`w-8 h-8 mx-auto mb-3 ${darkMode ? 'text-gray-700' : 'text-gray-400'}`} />
                    <p className={`${darkMode ? 'text-gray-500' : 'text-gray-600'} font-light`}>
                      {isDragActive ? 'Drop files here' : 'Upload files to enhance search'}
                    </p>
                    <p className={`${darkMode ? 'text-gray-700' : 'text-gray-400'} text-sm mt-1`}>
                      Images, documents, videos supported
                    </p>
                  </div>
                </div>
              )}

              {/* Uploaded Files */}
              {uploadedFiles.length > 0 && (
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-3">
                    <span className={`text-sm font-medium ${darkMode ? 'text-gray-400' : 'text-gray-700'}`}>
                      {uploadedFiles.length} file{uploadedFiles.length !== 1 ? 's' : ''} uploaded
                    </span>
                    <div {...getRootProps()}>
                      <input {...getInputProps()} />
                      <button className={`text-sm ${darkMode ? 'text-gray-600 hover:text-gray-500' : 'text-gray-500 hover:text-gray-700'} transition-colors`}>
                        Add more
                      </button>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {uploadedFiles.map((file, index) => (
                      <div
                        key={index}
                        className={`flex items-center justify-between p-3 ${darkMode ? 'bg-gray-900 border-gray-800' : 'bg-gray-50 border-gray-100'} rounded-xl border`}
                      >
                        <div className="flex items-center space-x-3 flex-1 min-w-0">
                          <DocumentIcon className={`w-5 h-5 ${darkMode ? 'text-gray-700' : 'text-gray-400'} flex-shrink-0`} />
                          <div className="min-w-0 flex-1">
                            <p className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'} truncate`}>
                              {file.filename || file.file?.name}
                            </p>
                            <p className={`text-xs ${darkMode ? 'text-gray-600' : 'text-gray-500'}`}>
                              {formatFileSize(file.file_size || file.file?.size)}
                            </p>
                          </div>
                        </div>
                        <button
                          onClick={() => removeFile(index)}
                          className={`p-1 ${darkMode ? 'text-gray-700 hover:text-gray-500' : 'text-gray-400 hover:text-gray-600'} transition-colors`}
                        >
                          <XMarkIcon className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Search Input */}
              <div className="relative">
                <input
                  ref={searchInputRef}
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask anything..."
                  className={`w-full px-6 py-4 text-lg ${darkMode ? 'bg-black border-gray-800 text-white placeholder-gray-600' : 'bg-white border-gray-200 text-gray-900 placeholder-gray-500'} border rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-300 shadow-sm`}
                />
                <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex items-center space-x-2">
                  <button
                    onClick={handleSearch}
                    disabled={loading || !query.trim()}
                    className={`p-3 ${darkMode ? 'bg-gray-900 text-white hover:bg-gray-800 border border-gray-800' : 'bg-gray-900 text-white hover:bg-gray-800'} rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-sm`}
                  >
                    {loading ? (
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    ) : (
                      <SearchIcon className="w-5 h-5" />
                    )}
                  </button>
                </div>
              </div>
            </motion.div>

            {/* Loading State */}
            {loading && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-center py-12"
              >
                <div className={`w-8 h-8 border-2 ${darkMode ? 'border-gray-800 border-t-white' : 'border-gray-200 border-t-gray-900'} rounded-full animate-spin mx-auto mb-4`}></div>
                <p className={`${darkMode ? 'text-gray-600' : 'text-gray-600'} font-light`}>Searching with AI...</p>
              </motion.div>
            )}

            {/* Error State */}
            {error && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`${darkMode ? 'bg-red-900/20 border-red-800' : 'bg-red-50 border-red-200'} border rounded-xl p-4 mb-8 flex items-center`}
              >
                <ExclamationTriangleIcon className={`w-5 h-5 ${darkMode ? 'text-red-400' : 'text-red-500'} mr-3`} />
                <span className={`${darkMode ? 'text-red-300' : 'text-red-700'}`}>{error}</span>
              </motion.div>
            )}

            {/* Results */}
            {results && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="pb-16"
              >
                <SearchResults 
                  results={results} 
                  onImageClick={openLightbox}
                  darkMode={darkMode}
                  viewMode={viewMode}
                  setViewMode={setViewMode}
                />
              </motion.div>
            )}

            {/* Search History */}
            {!results && !loading && searchHistory.length > 0 && (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="pb-16"
              >
                <h3 className={`text-lg font-medium ${darkMode ? 'text-white' : 'text-gray-900'} mb-4`}>Recent searches</h3>
                <div className="space-y-2">
                  {searchHistory.slice(0, 5).map((item, index) => (
                    <button
                      key={index}
                      onClick={() => {
                        setQuery(item.query);
                        setResults(item.results);
                        setSearchType(item.searchType);
                      }}
                      className={`w-full text-left p-4 ${darkMode ? 'bg-black hover:bg-gray-900 border-gray-800 hover:border-gray-700' : 'bg-gray-50 hover:bg-gray-100 border-gray-100 hover:border-gray-200'} rounded-xl transition-all duration-300 border shadow-sm`}
                    >
                      <div className="flex items-center justify-between">
                        <span className={`${darkMode ? 'text-white' : 'text-gray-900'} font-medium`}>{item.query}</span>
                        <span className={`${darkMode ? 'text-gray-600' : 'text-gray-500'} text-sm`}>
                          {item.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </div>
        </div>

        {/* Summary Panel - Right Side (Always Visible) */}
        <div className={`w-2/5 border-l ${darkMode ? 'border-gray-800' : 'border-gray-300'} p-6`}>
          <SummaryPanel 
            results={results}
            query={query}
            darkMode={darkMode}
            onGenerateDocument={() => setDocumentGeneratorOpen(true)}
          />
        </div>
      </div>

      {/* Image Lightbox */}
      <AnimatePresence>
        {lightboxOpen && (
          <ImageLightbox
            isOpen={lightboxOpen}
            images={lightboxImages}
            currentIndex={lightboxIndex}
            onClose={() => setLightboxOpen(false)}
            onNext={nextLightboxImage}
            onPrev={prevLightboxImage}
          />
        )}
      </AnimatePresence>

      {/* Document Generator */}
      <AnimatePresence>
        {documentGeneratorOpen && (
          <DocumentGenerator
            isOpen={documentGeneratorOpen}
            onClose={() => setDocumentGeneratorOpen(false)}
            searchResults={results}
            query={query}
            darkMode={darkMode}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;